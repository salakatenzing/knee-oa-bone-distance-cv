function averageThickness = AverageCartilageThickness(image)
    BinaryImage = image;
    
    se=strel('square',5);
    BinaryImage=imerode(BinaryImage,se);
    BinaryImage(1:135,:)= 255;
    
    BinaryImage(1:90,:)= 255;

    % step 4
    % labels each of the component to a specfic label/value to identify it
    labeledOutputImage = logical(BinaryImage);
    regions = regionprops(labeledOutputImage, 'Area', 'PixelIdxList', 'BoundingBox');

    % step 5
    % idenfiying the femur and tibia
    [~, sortedIdx] = sort([regions.Area], 'descend');
    if length(sortedIdx) < 2
        disp('Warning: Not enough regions detected in image ');
    else
        femurRegion = regions(sortedIdx(1)).PixelIdxList;
        tibiaRegion = regions(sortedIdx(2)).PixelIdxList;

        femurMask = zeros(size(BinaryImage));
        femurMask(femurRegion) = 1;
        tibiaMask = zeros(size(BinaryImage));
        tibiaMask(tibiaRegion) = 1;

        %step 6
        % Extract Boundaries
        femurBoundary = bwboundaries(femurMask);
        tibiaBoundary = bwboundaries(tibiaMask);

        if ~isempty(femurBoundary) && ~isempty(tibiaBoundary)
            % step 7
            minYFemur = max(femurBoundary{1}(:, 1)); 
            maxYTibia = minYFemur;

            % step 8 defining the margin and the boundaries
            verticalMargin = 7;
            horizontalMargin = 15;
            % selecting points for femur
            femurStartIdx = find(femurBoundary{1}(:,1) >= minYFemur - verticalMargin, 1) - horizontalMargin;
            femurEndIdx = find(femurBoundary{1}(:,1) >= minYFemur - verticalMargin, 1, 'last') + horizontalMargin;
            femurBoundaryExtended = femurBoundary{1}(max(femurStartIdx,1):min(femurEndIdx,size(femurBoundary{1},1)), :);
            % selecting points for tibia
            tibiaStartIdx = find(tibiaBoundary{1}(:,1) >= maxYTibia + verticalMargin, 1) - horizontalMargin;
            tibiaEndIdx = find(tibiaBoundary{1}(:,1) >= maxYTibia + verticalMargin, 1, 'last') + horizontalMargin;
            tibiaBoundaryExtended = flipud(tibiaBoundary{1}(max(tibiaStartIdx,1):min(tibiaEndIdx,size(tibiaBoundary{1},1)), :));

            %step 10
            % calculating distance
            distancesMatrix = pdist2(femurBoundaryExtended, tibiaBoundaryExtended);
            minDistances = min(distancesMatrix, [], 2);
            averageThickness = mean(minDistances);
            %overallMinDistance = min(minDistances);

          else
            % If boundaries are not found, display a warning
            disp(['Warning: Could not detect boundaries in image ', slice_filename]);
        end
    end
end
