function normalised_vector = normalisation_center(data)
    % normalisation_center
    %   returns the normalised vector after performing transformations like
    %   making the center (0, 0, 0) and then dividing the points by maximum
    %   distance
    %
    %   Example:
    %       result = normalisation_center(nxm);
    %
    %       n = number of rows
    %       m = number of cols
    %       result is a normalised vector of nxm

    
    % Dropping any NaN or empty row values as the data might contain missing
    % points as well.
    data = rmmissing(data);

    % Subtracting mean from every datapoint
    data = data - mean(data, 1);

    % Subtracting mean from every datapoint
    max_distance = max(sqrt(sum(data.^2, 2)));

    % Dividing max_distance from every datapoint and adding a small value
    % to handle division by zero
    data = data/(max_distance + 1e-8);
    normalised_vector = data;
end