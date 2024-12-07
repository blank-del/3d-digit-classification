function processed_data = process_data(data)
    % process_data
    %   returns the processed data after performing a certain
    %   transformations steps in a specific order with pre-fixed arguments
    %
    %   Example:
    %       result = normalisation_center(nxm);
    %
    %       n = number of rows
    %       m = number of cols
    %       result is a normalised vector of Nxm where N is interpolated
    %       data points to return
    [uniqueData, ia, ic] = unique(data, 'rows', 'stable');
    processed_data = interpolate_by_distance(uniqueData, 30);
    processed_data = smoothing(processed_data, 'savgol_filter',  5);
    processed_data = smoothing(processed_data, 'rolling', 3);
    processed_data = normalisation_center(processed_data);
    processed_data = smoothing(processed_data, 'savgol_filter', 5);
    processed_data = interpolate_by_distance(processed_data, 300);

end