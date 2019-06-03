#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;


int Sum(vector<int>& data) {
    vector<int> dp(data.size(), 0);
    dp[0] = data[0];
    for (int i = 1; i < data.size(); ++ i) {
        dp[i] = data[i];
        for (int j = 0; j < i; ++ j) {
            if (data[i] > data[j]) {
                dp[i] = max(dp[i], dp[j] + data[i]);
            }
        }
    }

    return dp[data.size() - 1];
}

int main() {
    vector<int> data = {5,1,3,4,9,7,6,8};
    cout << Sum(data) << endl;

    return 0;
}
