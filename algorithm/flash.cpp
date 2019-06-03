#include <iostream>
#include <vector>
#include <string>

using namespace std;

int get_sum(int len) {
    return (len + 1) * len / 2;
}

string get_code(int k) {
    string s;

    // edge cases
    if (k == 0) {
        return s;
    }
    
    // general cases
    s = "1";
    int len = 2;
    while (get_sum(len) < k) {
        if (len % 10 == 0) {
            s += "101";
        }else {
            s += to_string(len);
        }
        ++ len;
    }
    if (len % 10 == 0) {
        s += "101";
    }else {
        s += to_string(len);
    }

    return s;
}

double prob(int w, int b, bool take_white) {
    if (take_white) {
        return (w * w + 2 * w * b) * 1.0 / ((w + b) * (w + b));
    }else {
        return b * b * 1.0 / ((w + b) * (w + b));
    }
}

void prob_of_last_white_ball() {
    /// prob of white & black ball
    int W = 8, B = 7;
    cout << "please input W: ";
    cin >> W;
    cout << "please input B: ";
    cin >> B;

    vector<vector<double>> dp(W + 2, vector<double>(B + 2, 0));
    dp[W][B] = 1.0;
    for (int w = W; w >= 1; -- w) {
        for (int b = B - 1; b >= 0; -- b) {
            dp[w][b] = prob(w+1, b, true) * dp[w + 1][b] + 
                prob(w, b+1, false) * dp[w][b + 1];
        }
    }
    cout << dp[1][0] << endl;
}

vector<vector<int>> mat_mul(const vector<vector<int>>& A, 
        const vector<vector<int>>& B) {
    int n = A.size();
    vector<vector<int>> C(n, vector<int>(n, 0));
    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < n; ++ j) {
            for (int k = 0; k < n; ++ k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

void fast_matrix_exponent() {
    vector<vector<int>> mat = {{1,2,3}, {4,5,6}, {7,8,9}};
    int exp = 1;
    
    cout << "please input the exponent of the matirx: ";
    cin >> exp;

    // fast matrix exp
    vector<vector<int>> res = {{1,0,0}, {0,1,0}, {0,0,1}};
    if (exp) {
        vector<vector<int>> factor = mat;
        if (exp & 1) {
            res = mat;
        }
        exp >>= 1;
        while (exp) {
            factor = mat_mul(factor, factor);
            if (exp & 1) {
                res = mat_mul(factor, res);
            }
            exp >>= 1;
        }
    }

    // output 
    cout << "after the exponent, the matrix is: " << endl;
    for (auto v : res) {
        for (auto n : v) {
            cout << n << ",";
        }
        cout << endl;
    }
}

int main() {
    /// prob_of_last_white_ball();
    fast_matrix_exponent();

    return 0;
}
