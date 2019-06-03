// Complete the meet function below.
vector<int> meet(vector<string> wizards) {
    vector<int> res(1, 0);
    
    // edge cases
    if (wizards.size() == 2) {
        if (wizards[0].empty()) {
            return res;
        }
    }
    
    // general cases
    // turn matrix
    int N = wizards.size();
    vector<vector<int>> weight(N, vector<int>(N, INT_MAX));
    for (int i = 0; i < N; ++ i) {
        istringstream is(wizards[i]);
        int dst;
        while (is >> dst) {
            if (i == 0) {
                weight[i][dst] = 0;
            }else {
                weight[i][dst] = (dst - i) * (dst - i);
            }
        }
    }
    
    // Dijkstra
    vector<int> dist(N, INT_MAX);
    vector<int> prev(N, -1);
    vector<bool> visited(N, false);
    int src = 0;
    for (int i = 0; i < N; ++ i) {
        dist[i] = weight[src][i];
        if (dist[i] == INT_MAX) {
            prev[i] = -1;
        }else {
            prev[i] = src;
        }
    }
    dist[src] = 0;
    visited[src] = true;
    
    for (int i = 1; i < N; ++ i) {
        int min_dist = INT_MAX;
        // find passed min
        int u = src;
        for (int j = 0; j < N; ++ j) {
            if (!visited[j] && dist[j] < min_dist) {
                u = j;
                min_dist = dist[j];
            }
        }
        visited[u] = true;
        
        // update dist
        for (int j = 0; j < N; ++ j) {
            if (!visited[j] && weight[u][j] < INT_MAX) {
                if (dist[u] + weight[u][j] < dist[j]) {
                    dist[j] = dist[u] + weight[u][j];
                    prev[j] = u;
                }
            }
        }
    }
    
    if (dist[N - 1] == INT_MAX) {
        return res;
    }
    
    res.pop_back();
    res.push_back(N - 1);
    int cur = N - 1;
    while (cur != 0) {
        res.push_back(prev[cur]);
        cur = prev[cur];
    }
    
    return vector<int>(res.rbegin(), res.rend());
}

