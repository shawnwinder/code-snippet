// from Airbnb 1 hour online test
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <climits>

using namespace std;

vector<int> meet(vector<string> wizards) {
    vector<int> res(1, 0);

    // edge cases
    if (wizards.size() < 2) {
        return res;
    }
    if (wizards.size() == 2 && wizards[0].empty()) {
        return res;
    }

    // general cases
    // get weight matrix
    // int N = wizards.size();
    // vector<vector<int>> weight(N, vector<int>(N, INT_MAX));
    // for (int i = 0; i < N; ++ i) {
    //     istringstream is(wizards[i]);
    //     int dst;
    //     while (is >> dst) {
    //         if (i == 0) {
    //             weight[i][dst] = 0;
    //         }else {
    //             weight[i][dst] = (dst - i) * (dst - i);
    //         }
    //     }
    // }

    int N = 6;
    vector<vector<int>> weight(N, vector<int>(N, INT_MAX));
    for (auto s : wizards) {
        istringstream is(s);
        int src, dst, d; 
        is >> src >> dst >> d;
        weight[src][dst] = d;
    }

    // Djikstra
    // init
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
        // find current shortest
        int min_dist = INT_MAX;
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
            // if (!visited[j] && dist[u] + weight[u][j] < dist[j]) {   // overflow!
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
       
    cout << "shortest path form src to dst (0, N-1) is: ";
    for (auto n : dist) cout << n << ",";
    cout << endl;

    res.pop_back();
    int i = N - 1;
    while (i != src) {
        res.push_back(i);
        i = prev[i];
    }
    res.push_back(src);

    return vector<int>(res.rbegin(), res.rend());
}

int main() {
    vector<string> wizards = {
        "0 1 7",
        "0 5 14",
        "0 2 9",
        "1 2 10",
        "1 3 15",
        "2 3 11",
        "2 5 2",
        "3 4 6",
        "4 5 9",
        "1 0 7",
        "5 0 14",
        "2 0 9",
        "2 1 10",
        "3 1 15",
        "3 2 11",
        "5 2 2",
        "4 3 6",
        "5 4 9",
    };

    vector<int> path = meet(wizards);

    for (auto n : path) {
        cout << n << ",";
    }
    cout << endl;

    return 0;
}




