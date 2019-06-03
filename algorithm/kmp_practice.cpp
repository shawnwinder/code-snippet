#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <utility>

using namespace std;

int violent_match(const string& str, const string& ptn) {
    int i = 0, j = 0;
    while (i < str.size() && j < ptn.size()) {
        if (str[i] == ptn[j]) {
            ++ i;
            ++ j;
        }else {
            i = i - j + 1;
            j = 0;
        }
    }
    if (j == ptn.size()) {
        return i - j;
    }else {
        return -1;
    }
}

vector<int> get_next(const string& ptn) {
    vector<int> next(ptn.size(), -1);
    int idx = 0;
    int len = next[idx];
    while (idx < ptn.size() - 1) {
        if (len == -1 || ptn[idx] == ptn[len]){
            ++ idx;
            ++ len;
            if (ptn[idx] != ptn[len]) {
                next[idx] = len;
            }else {
                next[idx] = next[len];
            }
        }else {
            len = next[len];
        }
    }

    return next;
}

int kmp(const string& str, const string& ptn) {
    int len_str = str.size();
    int len_ptn = ptn.size();
    int i = 0, j = 0;
    auto next = get_next(ptn);
    while (i < len_str && j < len_ptn) {
        if (j == -1 || str[i] == ptn[j]) {
            ++ i;
            ++ j;
        }else {
            j = next[j];
        }
    }
    if (j == len_ptn) {
        return i - j;
    }else {
        return -1;
    }
}

int main() {
    const string str = "BBC ABCDAB ABCDABCDABDE";
    const string ptn = "ABCDABD";
    // const string ptn = "ABAB";
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
