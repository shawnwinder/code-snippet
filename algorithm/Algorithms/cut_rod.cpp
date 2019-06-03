#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int cut_rod(const vector<int>& price, int len) {
    vector<int> dp(len + 1, 0);

    // bottom-up way
    for (int i = 1; i <= len; ++ i) {
        for (int j = 1; j <= i; ++ j) {
            dp[i] = max(dp[i], price[j] + dp[i - j]);
        }
    }

    return dp[len];
}

int main() {
    vector<int> price = {0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30};

    while (true) {
        int len = 10;
        cout << "Please input the length of the rod: " << endl;
        cin >> len;
        cout << "The max profit of cutting the rod is : " << cut_rod(price, len) << endl;
        cout << endl;
    }

    return 0;
}
