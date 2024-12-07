function interpolated = interpolate_by_distance(data, num_points)
    % interpolate_by_distance
    %   returns the interpolated vector with num_points mentioned in the
    %   function input, interpolated points are calculated using the
    %   cumulative distance between points
    %
    %   Example:
    %       interpolated = interpolate_by_distance(data, num_points);
    %       
    %       data (n x m)
    %       n = number of rows
    %       m = number of cols
    %       num_points = number of points to interpolate
    %       result is a vector of num_points x m

    
    % Calculate cumulative distance
    % Compute differences along the first dimension
    diff_trajectory = diff(data, 1, 1);
    % Euclidean distance between consecutive points
    distances = sqrt(sum(diff_trajectory.^2, 2)); 
    % Cumulative distance, starting from 0
    cumulative_distances = [0; cumsum(distances)];  
    
    % Total distance
    total_distance = cumulative_distances(end);
    
    % Resample based on cumulative distance
    new_distances = linspace(0, total_distance, num_points)';
    
    % Linear interpolation of the trajectory based on cumulative distance
    interpolated = interp1(cumulative_distances, data, new_distances, 'linear');
end