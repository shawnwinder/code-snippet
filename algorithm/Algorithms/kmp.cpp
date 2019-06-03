#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>

using namespace std;

int violent_match(const string& str, const string& ptn) {
    int i = 0, j = 0;
    while (i < str.size() && j < ptn.size()) {
        if (str[i] == ptn[j]) {
            ++ i;
            ++ j;
        }else {
            i = i - (j - 1);
            j = 0;
        }
    }
    if (j == ptn.size()) {
        return i - (j - 1);
    }else {
        return -1;
    }
}

// recursively compute next 
vector<int> get_next(const string& ptn) {
    vector<int> next(ptn.size(), -1);
    int pre = next[0];
    // must be 0
    int idx = 0; 
    while (idx < ptn.size() - 1) {
        // ptn[pre] means the next 1 index defined by the longest equal prefix & suffix 
        // pre == -1 means the begin of the match or the match from the begin of ptn
        if (pre == -1 || ptn[pre] == ptn[idx]) {
            ++ pre;
            ++ idx;
            // next[idx] = pre;

            // fast version
            if (ptn[pre] != ptn[idx]) {
                next[idx] = pre;
            }else {
                next[idx] = next[pre];
            }
        }else {
            pre = next[pre];
        }
    }

    return next;
}

// kmp using next
int kmp(const string& str, const string& ptn) {
    int i = 0, j = 0;
    vector<int> next = get_next(ptn);
    for (auto n : next) cout << n << ",";
    cout << endl;

    while (i < str.size() && j < (int)ptn.size()) {
        if (j == -1 || str[i] == ptn[j]) {
            ++ i;
            ++ j;
        }else {
            j = next[j];
        }
    }
    if (j == ptn.size()) {
        return i - (j - 1);
    }else {
        return -1;
    }
}

int main() {
    const string str = "BBC ABCDAB ABCDABCDABDE";
    // const string ptn = "ABCDABD";
    const string ptn = "ABAB";
    // const string ptn = "ABCDAXD";
    // int ret = violent_match(str, ptn);
    int ret = kmp(str, ptn);
    if (ret == -1) {
        cout << "\"" << ptn << "\"" << " is not in " << "\"" << str << "\"" << endl;
    }else {
        cout << "Found " << "\"" << ptn << "\"" << " in " << ret << " of " << "\"" << str << "\"" << endl;
    }

    return 0;
}
