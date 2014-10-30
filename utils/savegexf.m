function savegexf(filename, thetaNodeArray, thetaEdgesArray, words, numTopWords)
%SAVEGEXF Save PMRF(s) parameters as a graph in the GEXF format to open in Gephi
% Input:
% filename      Extension of ".gexf" will be added if needed
% thetaNodeArray K x 1 cell array of K node vectors
% thetaEdgesArray K x 1 cell array of K edge matrices
% words         Cell array of words
% numTopWords   Number of words with the highest node weight to keep even
%    if they do not have any edges.  If this is set to the number of words,
%    then all words will be saved. (Default: 0, i.e. only nodes with edges 
%    are saved)
%
% See http://gexf.net/format/ for GEXF format and http://gephi.github.io/
% for Gephi software
%
% function savegexf(filename, thetaNodeArray, thetaEdgesArray, words, numTopWords)
if(nargin < 5); numTopWords = 0; end;

% Add extension if needed
if(~strcmp('.gexf', filename((end-4):end)))
    filename = [filename '.gexf'];        
end
fid = fopen(filename, 'W+');

level = 0;
xml('<?xml version="1.0" encoding="UTF-8"?>');
level = 0; % Reset level for xml output indentation
xml('<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd" version="1.2">');

xml(sprintf('<meta lastmodifieddate="%s">', datestr(date, 29)));
xml('<creator>David Inouye</creator>');
xml('<description>Graphs over words</description>');
xml('</meta>');

xml('<graph defaultedgetype="undirected">');
xml('<attributes class="node">');
xml('<attribute id="1" title="topic" type="integer"/>');
xml('<attribute id="2" title="expTheta" type="float"/>');
xml('<attribute id="3" title="theta" type="float"/>');
xml('</attributes>');
xml('<attributes class="edge">');
xml('<attribute id="1" title="expTheta" type="float"/>');
xml('<attribute id="2" title="theta" type="float"/>');
xml('</attributes>');

xml('<nodes>');

%% Initialize nodes
nodesSize = -1;
nodes = [];
add_node();
for j = 1:size(thetaNodeArray,1)
    % Find top word indices
    theta = thetaEdgesArray{j};
    thetaNode = thetaNodeArray{j}; % Theta values of nodes
    [~, topWordIdx] = sort(diag(theta),1,'descend');
    
    for i = 1:length(words)
        % Skip if not connected unless top word
        numPosEdges = sum(theta(i,:) > 0);
        wordRank = find(i == topWordIdx);
        if(numPosEdges == 0 && wordRank > numTopWords) 
            continue;
        end
        
        add_node([j i]);
        xml(sprintf('<node id="%s" label="%s">', node_id(j,i), words{i}) );
        xml('<attvalues>');
        xml(sprintf('<attvalue for="1" value="%d"/>', j ) );
        xml(sprintf('<attvalue for="2" value="%f"/>', exp(thetaNode(i)) ) );
        xml(sprintf('<attvalue for="3" value="%f"/>', thetaNode(i) ) );
        xml('</attvalues>');
        xml('</node>');
    end
end
xml('</nodes>');

xml('<edges>');
edgeId = 0;
nodes = get_nodes(); % Get trimmed nodes
for j = 1:length(thetaNodeArray)
    theta = thetaEdgesArray{j};
    for i = 1:length(words)
        for i2 = 1:length(words)
            if(i2 >= i ...              % Only deal with lower half of theta
                || sum(nodes(:,1) == j & nodes(:,2) == i) < 1 ...  % Skip nodes that were not added to the node list
                || sum(nodes(:,1) == j & nodes(:,2) == i2) < 1 ... % Skip nodes that were not added to the node list
                || theta(i,i2) <= 0)     % Ignore negative edges for now
                continue;               
            end
            if(sum(nodes(:,1) == j & nodes(:,2) == i) > 1)
                warning('More than 1 node matched');
            end
            
            edgeId = edgeId+1;
            xml(sprintf('<edge id="%d" source="%s" target="%s" weight="%f">', ...
                edgeId, node_id(j,i), node_id(j,i2), full(theta(i,i2)) ) );
            xml('<attvalues>');
            xml(sprintf('<attvalue for="1" value="%f"/>', exp(full(theta(i,i2))) ) );
            xml(sprintf('<attvalue for="2" value="%f"/>', full(theta(i,i2)) ) );
            xml('</attvalues>');    
            xml('</edge>');
            
        end
    end
end
xml('</edges>');

xml('</graph>');
xml('</gexf>');

fclose(fid);

    function nid = node_id(topicId, wordId)
        nid = sprintf('%d.%d', topicId, wordId);
    end

    function add_node(node)
        if(nargin < 1)
            nodes = zeros(1000,2);
            nodesSize = 0;
            return;
        end
        nodesSize = nodesSize + 1;
        if(nodesSize > size(nodes, 1)) 
            % Double the size of the array
            nodesNew = zeros(size(nodes,1)*2 ,2);
            nodesNew(1:size(nodes,1), :) = nodes;
            nodes = nodesNew;
        end
        nodes(nodesSize, :) = node;
        return;
    end

    function [nodesTrimmed] = get_nodes()
        nodesTrimmed = nodes(1:nodesSize, :);
        return;
    end

    function xml(tag)
        if(tag(2) == '/')
            level = level - 1;
        end
        if(tag(1) ~= '<'); error('Not proper tag'); end;
        tabString = repmat('\t', 1, level);
        fprintf(fid, [tabString '%s\n'], tag);
        if(tag((end-1)) ~= '/' && isempty(strfind(tag, '</')))
            level = level + 1;
        end
    end

end