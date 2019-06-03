#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int knapsack(const vector<int>& weight, const vector<int>& value, int V) {
    int N = weight.size() - 1;
    vector<int> dp(V + 1, 0);
    for (int i = 1; i <= N; ++ i) {
        for (int j = V; j >= 0; -- j) {
            if (j >= weight[i]) {
                dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);
            }
        }
    }

    return dp[V];
}

int main() {
    int T;
    cin >> T;
    for (int t = 0; t < T; ++ t) {
        int N, V;
        cin >> N >> V;
        vector<int> weight(N + 1, 0);
        vector<int> value(N + 1, 0);
        for (int i = 1; i <= N; ++ i) {
            cin >> value[i];
        }
        for (int i = 1; i <= N; ++ i) {
            cin >> weight[i];
        }

        cout << knapsack(weight, value, V) << endl;
    }

    return 0;
}
