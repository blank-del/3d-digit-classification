function smooth_data = smoothing (data, type, window_length)
    % smoothing
    %   returns the smoothed vector after performing transformations like
    %   either the curve fitting of order '2' using savgol or using the rolling mean method
    %
    %   Example:
    %       result = normalisation_center(data, type, window_length);
    %       
    %       data = nxm matrix
    %       n = number of rows
    %       m = number of cols
    %       type = type of filter that should be used
    %       window_length = window lenght to be used for transformation
    %
    %       result is a smoothed out vector of nxm (might contain NaN's)
    smooth_data = zeros(size(data));
    % to store the data points after smoothing is applied
    if type == "savgol_filter"
        % this type of filter is applied to keep the shape as close to the
        % original one by modelling a curve of order x, over the window
        % length w, the higher the order the more closer it will be the
        % original shape
        smooth_data(:, 1) = sgolayfilt(data(:, 1) , 2, window_length);
        smooth_data(:, 2) = sgolayfilt(data(:, 2) , 2, window_length);
        smooth_data(:, 3) = sgolayfilt(data(:, 3) , 2, window_length);
    elseif type == "rolling"
        % this type of filter is applied to keep smooth out the edges and
        % the padding of NaN is applied so that for end points the data
        % doesn't have enough points
        padding_size = floor(window_length / 2);
        smooth_data(:, 1) = [NaN(padding_size, 1); movmean(data(:, 1), window_length, 'Endpoints', 'discard'); NaN(padding_size, 1)];
        smooth_data(:, 2) = [NaN(padding_size, 1); movmean(data(:, 2), window_length, 'Endpoints', 'discard'); NaN(padding_size, 1)];
        smooth_data(:, 3) = [NaN(padding_size, 1); movmean(data(:, 3), window_length, 'Endpoints', 'discard'); NaN(padding_size, 1)];
    end 
end