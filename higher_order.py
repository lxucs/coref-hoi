import torch
import torch.nn as nn
import util


def attended_antecedent(top_span_emb, top_antecedent_emb, top_antecedent_scores, device):
    num_top_spans = top_span_emb.shape[0]
    top_antecedent_weights = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=1)
    top_antecedent_weights = nn.functional.softmax(top_antecedent_weights, dim=1)
    top_antecedent_emb = torch.cat([torch.unsqueeze(top_span_emb, 1), top_antecedent_emb], dim=1)
    refined_span_emb = torch.sum(torch.unsqueeze(top_antecedent_weights, 2) * top_antecedent_emb, dim=1)  # [num top spans, span emb size]
    return refined_span_emb


def max_antecedent(top_span_emb, top_antecedent_emb, top_antecedent_scores, device):
    num_top_spans = top_span_emb.shape[0]
    top_antecedent_weights = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=1)
    top_antecedent_emb = torch.cat([torch.unsqueeze(top_span_emb, 1), top_antecedent_emb], dim=1)
    max_antecedent_idx = torch.argmax(top_antecedent_weights, dim=1, keepdim=True)
    refined_span_emb = util.batch_select(top_antecedent_emb, max_antecedent_idx, device=device).squeeze(1)  # [num top spans, span emb size]
    return refined_span_emb


def entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx, top_antecedent_scores, device):
    # Use TF implementation in another repo
    pass

def span_clustering(top_span_emb, top_antecedent_idx, top_antecedent_scores, span_attn_ffnn, device):
    # Get predicted antecedents
    num_top_spans, max_top_antecedents = top_antecedent_idx.shape[0], top_antecedent_idx.shape[1]
    predicted_antecedents = []
    top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_antecedent_scores], dim=1)
    for i, idx in enumerate((torch.argmax(top_antecedent_scores, axis=1) - 1).tolist()):
        if idx < 0:
            predicted_antecedents.append(-1)
        else:
            predicted_antecedents.append(top_antecedent_idx[i, idx].item())
    # Get predicted clusters
    predicted_clusters = []
    span_to_cluster_id = [-1] * num_top_spans
    for i, predicted_idx in enumerate(predicted_antecedents):
        if predicted_idx < 0:
            continue
        assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
        # Check antecedent's cluster
        antecedent_cluster_id = span_to_cluster_id[predicted_idx]
        if antecedent_cluster_id == -1:
            antecedent_cluster_id = len(predicted_clusters)
            predicted_clusters.append([predicted_idx])
            span_to_cluster_id[predicted_idx] = antecedent_cluster_id
        # Add mention to cluster
        predicted_clusters[antecedent_cluster_id].append(i)
        span_to_cluster_id[i] = antecedent_cluster_id
    if len(predicted_clusters) == 0:
        return top_span_emb

    # Pad clusters
    max_cluster_size = max([len(c) for c in predicted_clusters])
    cluster_sizes = []
    for cluster in predicted_clusters:
        cluster_sizes.append(len(cluster))
        cluster += [0] * (max_cluster_size - len(cluster))
    predicted_clusters_mask = torch.arange(0, max_cluster_size, device=device).repeat(len(predicted_clusters), 1)
    predicted_clusters_mask = predicted_clusters_mask < torch.tensor(cluster_sizes, device=device).unsqueeze(1)  # [num clusters, max cluster size]
    # Get cluster repr
    predicted_clusters = torch.tensor(predicted_clusters, device=device)
    cluster_emb = top_span_emb[predicted_clusters]  # [num clusters, max cluster size, emb size]
    span_attn = torch.squeeze(span_attn_ffnn(cluster_emb), 2)
    span_attn += torch.log(predicted_clusters_mask.to(torch.float))
    span_attn = nn.functional.softmax(span_attn, dim=1)
    cluster_emb = torch.sum(cluster_emb * torch.unsqueeze(span_attn, 2), dim=1)  # [num clusters, emb size]
    # Get refined span
    refined_span_emb = []
    for i, cluster_idx in enumerate(span_to_cluster_id):
        if cluster_idx < 0:
            refined_span_emb.append(top_span_emb[i])
        else:
            refined_span_emb.append(cluster_emb[cluster_idx])
    refined_span_emb = torch.stack(refined_span_emb, dim=0)
    return refined_span_emb


def cluster_merging(top_span_emb, top_antecedent_idx, top_antecedent_scores, emb_cluster_size, cluster_score_ffnn, cluster_transform, dropout, device, reduce='mean', easy_cluster_first=False):
    num_top_spans, max_top_antecedents = top_antecedent_idx.shape[0], top_antecedent_idx.shape[1]
    span_emb_size = top_span_emb.shape[-1]
    max_num_clusters = num_top_spans

    span_to_cluster_id = torch.zeros(num_top_spans, dtype=torch.long, device=device)  # id 0 as dummy cluster
    cluster_emb = torch.zeros(max_num_clusters, span_emb_size, dtype=torch.float, device=device)  # [max num clusters, emb size]
    num_clusters = 1  # dummy cluster
    cluster_sizes = torch.ones(max_num_clusters, dtype=torch.long, device=device)

    merge_order = torch.arange(0, num_top_spans)
    if easy_cluster_first:
        max_antecedent_scores, _ = torch.max(top_antecedent_scores, dim=1)
        merge_order = torch.argsort(max_antecedent_scores, descending=True)
    cluster_merging_scores = [None] * num_top_spans

    for i in merge_order.tolist():
        # Get cluster scores
        antecedent_cluster_idx = span_to_cluster_id[top_antecedent_idx[i]]
        antecedent_cluster_emb = cluster_emb[antecedent_cluster_idx]
        # antecedent_cluster_emb = dropout(cluster_transform(antecedent_cluster_emb))

        antecedent_cluster_size = cluster_sizes[antecedent_cluster_idx]
        antecedent_cluster_size = util.bucket_distance(antecedent_cluster_size)
        cluster_size_emb = dropout(emb_cluster_size(antecedent_cluster_size))

        span_emb = top_span_emb[i].unsqueeze(0).repeat(max_top_antecedents, 1)
        similarity_emb = span_emb * antecedent_cluster_emb
        pair_emb = torch.cat([span_emb, antecedent_cluster_emb, similarity_emb, cluster_size_emb], dim=1)  # [max top antecedents, pair emb size]
        cluster_scores = torch.squeeze(cluster_score_ffnn(pair_emb), 1)
        cluster_scores_mask = (antecedent_cluster_idx > 0).to(torch.float)
        cluster_scores *= cluster_scores_mask
        cluster_merging_scores[i] = cluster_scores

        # Get predicted antecedent
        antecedent_scores = top_antecedent_scores[i] + cluster_scores
        max_score, max_score_idx = torch.max(antecedent_scores, dim=0)
        if max_score < 0:
            continue  # Dummy antecedent
        max_antecedent_idx = top_antecedent_idx[i, max_score_idx]

        if not easy_cluster_first:  # Always add span to antecedent's cluster
            # Create antecedent cluster if needed
            antecedent_cluster_id = span_to_cluster_id[max_antecedent_idx]
            if antecedent_cluster_id == 0:
                antecedent_cluster_id = num_clusters
                span_to_cluster_id[max_antecedent_idx] = antecedent_cluster_id
                cluster_emb[antecedent_cluster_id] = top_span_emb[max_antecedent_idx]
                num_clusters += 1
            # Add span to cluster
            span_to_cluster_id[i] = antecedent_cluster_id
            _merge_span_to_cluster(cluster_emb, cluster_sizes, antecedent_cluster_id, top_span_emb[i], reduce=reduce)
        else:  # current span can be in cluster already
            antecedent_cluster_id = span_to_cluster_id[max_antecedent_idx]
            curr_span_cluster_id = span_to_cluster_id[i]
            if antecedent_cluster_id > 0 and curr_span_cluster_id > 0:
                # Merge two clusters
                span_to_cluster_id[max_antecedent_idx] = curr_span_cluster_id
                _merge_clusters(cluster_emb, cluster_sizes, antecedent_cluster_id, curr_span_cluster_id, reduce=reduce)
            elif curr_span_cluster_id > 0:
                # Merge antecedent to span's cluster
                span_to_cluster_id[max_antecedent_idx] = curr_span_cluster_id
                _merge_span_to_cluster(cluster_emb, cluster_sizes, curr_span_cluster_id, top_span_emb[max_antecedent_idx], reduce=reduce)
            else:
                # Create antecedent cluster if needed
                if antecedent_cluster_id == 0:
                    antecedent_cluster_id = num_clusters
                    span_to_cluster_id[max_antecedent_idx] = antecedent_cluster_id
                    cluster_emb[antecedent_cluster_id] = top_span_emb[max_antecedent_idx]
                    num_clusters += 1
                # Add span to cluster
                span_to_cluster_id[i] = antecedent_cluster_id
                _merge_span_to_cluster(cluster_emb, cluster_sizes, antecedent_cluster_id, top_span_emb[i], reduce=reduce)

    cluster_merging_scores = torch.stack(cluster_merging_scores, dim=0)
    return cluster_merging_scores


def _merge_span_to_cluster(cluster_emb, cluster_sizes, cluster_to_merge_id, span_emb, reduce):
    cluster_size = cluster_sizes[cluster_to_merge_id].item()
    if reduce == 'mean':
        cluster_emb[cluster_to_merge_id] = (cluster_emb[cluster_to_merge_id] * cluster_size + span_emb) / (cluster_size + 1)
    elif reduce == 'max':
        cluster_emb[cluster_to_merge_id], _ = torch.max(torch.stack([cluster_emb[cluster_to_merge_id], span_emb]), dim=0)
    else:
        raise ValueError('reduce value is invalid: %s' % reduce)
    cluster_sizes[cluster_to_merge_id] += 1


def _merge_clusters(cluster_emb, cluster_sizes, cluster1_id, cluster2_id, reduce):
    """ Merge cluster1 to cluster2 """
    cluster1_size, cluster2_size = cluster_sizes[cluster1_id].item(), cluster_sizes[cluster2_id].item()
    if reduce == 'mean':
        cluster_emb[cluster2_id] = (cluster_emb[cluster1_id] * cluster1_size + cluster_emb[cluster2_id] * cluster2_size) / (cluster1_size + cluster2_size)
    elif reduce == 'max':
        cluster_emb[cluster2_id] = torch.max(cluster_emb[cluster1_id], cluster_emb[cluster2_id])
    else:
        raise ValueError('reduce value is invalid: %s' % reduce)
    cluster_sizes[cluster2_id] += cluster_sizes[cluster1_id]
