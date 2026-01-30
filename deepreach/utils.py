import torch
import os

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def line_segment_distances(line_segments1, line_segments2, threshold=1e-6):
    # line segment 1: (batch_size, num_links+1, D)
    # line segment 2: (batch_size, num_links+1, D)
    # output: (batch_size, num_links, num_links)    
    
    line_segment1_start = line_segments1[..., :-1,:]
    line_segment2_start = line_segments2[..., :-1,:]
    
    u = (line_segments1[..., 1:,:] - line_segment1_start)[..., :, None, :] # direction1: (batch_size, num_links, 1, D)
    v = (line_segments2[..., 1:,:] - line_segment2_start)[..., None, :, :] # direction2: (batch_size, 1, num_links, D)
    r = line_segment2_start[..., None, :, :] - line_segment1_start[..., :, None, :]
    
    ru = torch.sum(r * u, dim=-1)
    rv = torch.sum(r * v, dim=-1)
    uu = torch.sum(u * u, dim=-1)
    uv = torch.sum(u * v, dim=-1)
    vv = torch.sum(v * v, dim=-1)
    
    det = uu * vv - uv * uv
    s = torch.clamp((ru * vv - rv * uv) / det, 0.0, 1.0)
    t = torch.clamp((ru * uv - rv * uu) / det, 0.0, 1.0)
    mask = det < threshold
    s[mask] = torch.clamp(ru / uu, 0.0, 1.0)[mask]
    t[mask] = 0.
    
    S = torch.clamp((t*uv + ru)/uu, 0., 1.).unsqueeze(-1)
    T = torch.clamp((s*uv - rv)/vv, 0., 1.).unsqueeze(-1)
    p1 = line_segment1_start[..., :, None, :] + S * u 
    p2 = line_segment2_start[..., None, :, :] + T * v
    
    return torch.linalg.norm(p1 - p2, dim=-1).flatten(start_dim=-2,end_dim=-1)

def line_segment_distances_and_points(line_segments1, line_segments2, threshold=1e-6):
    # line segment 1: (batch_size, num_links+1, D)
    # line segment 2: (batch_size, num_links+1, D)
    # output: (batch_size, num_links, num_links)    
    
    line_segment1_start = line_segments1[..., :-1,:]
    line_segment2_start = line_segments2[..., :-1,:]
    
    u = (line_segments1[..., 1:,:] - line_segment1_start)[..., :, None, :] # direction1: (batch_size, num_links, 1, D)
    v = (line_segments2[..., 1:,:] - line_segment2_start)[..., None, :, :] # direction2: (batch_size, 1, num_links, D)
    r = line_segment2_start[..., None, :, :] - line_segment1_start[..., :, None, :]
    
    ru = torch.sum(r * u, dim=-1)
    rv = torch.sum(r * v, dim=-1)
    uu = torch.sum(u * u, dim=-1)
    uv = torch.sum(u * v, dim=-1)
    vv = torch.sum(v * v, dim=-1)
    
    det = uu * vv - uv * uv
    s = torch.clamp((ru * vv - rv * uv) / det, 0.0, 1.0)
    t = torch.clamp((ru * uv - rv * uu) / det, 0.0, 1.0)
    mask = det < threshold
    s[mask] = torch.clamp(ru / uu, 0.0, 1.0)[mask]
    t[mask] = 0.
    
    S = torch.clamp((t*uv + ru)/uu, 0., 1.).unsqueeze(-1)
    T = torch.clamp((s*uv - rv)/vv, 0., 1.).unsqueeze(-1)
    p1 = line_segment1_start[..., :, None, :] + S * u 
    p2 = line_segment2_start[..., None, :, :] + T * v
    
    return torch.linalg.norm(p1 - p2, dim=-1).flatten(start_dim=-2, end_dim=-1), p1, p2
    

def random_distance_test(num_tests=200):
        num_points_each_batch = 10
    
        torch.manual_seed(0)
        A = torch.rand(num_tests, num_points_each_batch, 3)
        B = torch.rand(num_tests, num_points_each_batch, 3)

        distances = line_segment_distances(line_segments1=A, line_segments2=B).view(-1)
        
        import cvxpy as cp
        counter = 0
        for i_batch in range(num_tests):
            for i_point_A in range(num_points_each_batch-1):
                line_segment1_start = A[i_batch, i_point_A].numpy()
                line_segment1_end = A[i_batch, i_point_A + 1].numpy()
                for i_point_B in range(num_points_each_batch-1):
                    line_segment2_start = B[i_batch, i_point_B].numpy()
                    line_segment2_end = B[i_batch, i_point_B + 1].numpy()
                                        
                    x = cp.Variable(2)
                    objective = cp.Minimize(cp.sum_squares(line_segment1_start + x[0] * (line_segment1_end - line_segment1_start) - line_segment2_start - x[1] * (line_segment2_end - line_segment2_start)))
                    constraints = [0 <= x, x <= 1]
                    prob = cp.Problem(objective, constraints)

                    # The optimal objective value is returned by `prob.solve()`.
                    result = prob.solve()
                    result = result ** 0.5
                    # print(f"Closed form distance: {distances[counter]}, optimization distance: {result}")
                    if torch.abs(distances[counter] - result) > 1e-3:
                        p1 = line_segment1_start + x.value[0] * (line_segment1_end - line_segment1_start)
                        p2 = line_segment2_start - x.value[1] * (line_segment2_end - line_segment2_start)
                        print(f"Line1 start: {line_segment1_start}, Line 2 start:{line_segment2_start}" )
                        print(f"Points from line 1:{p1}, Points from line 2:{p2}, x:{x.value}")
                    counter += 1