#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

string LCS(string sa, string sb) {
    vector<vector<int>> dp(sa.size() + 1, vector<int>(sb.size() + 1, 0));
    for (int i = 1; i <= sa.size(); ++ i) {
        for (int j = 1; j <= sb.size(); ++ j) {
            if (sa[i - 1] == sb[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }else {
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
            }
        }
    }

    int len = dp[sa.size()][sb.size()];
    string LCS(len, '0');
    int i = sa.size(), j = sb.size(), cur = len - 1;
    while (i >= 1 && j >= 1) {
        if (sa[i - 1] == sb[j - 1]) {
            LCS[cur--] = sa[i - 1];
            i -= 1;
            j -= 1;
        }else if (dp[i - 1][j] > dp[i][j - 1]) {
            -- i;
        }else {
            -- j;
        }
    }

    return LCS;
}

int LCS_length(string sa, string sb) {
    vector<vector<int>> dp(sa.size() + 1, vector<int>(sb.size() + 1, 0));
    for (int i = 1; i <= sa.size(); ++ i) {
        for (int j = 1; j <= sb.size(); ++ j) {
            if (sa[i - 1] == sb[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }else {
                dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
            }
        }
    }
    return dp[sa.size()][sb.size()];
}

int main() { 
    string sa = "125376"; 
    string sb = "2876";
    int LCS_len  = LCS_length(sa, sb);
    string LCS_str = LCS(sa, sb);

    cout << "LCS length: " << LCS_len << endl;
    cout << "LCS string: " << LCS_str << endl;

    return 0;
}



