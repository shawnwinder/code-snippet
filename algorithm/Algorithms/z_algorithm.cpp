#include <iostream>
#include <vector>
#include <string>

using namespace std;

vector<int> get_zarray(const string& s) {
    vector<int> zarr(s.size(), -1);
    int L, R;
    L = R = 0;
    for (int i = 1; i < (int)s.size(); ++ i) {
        if (i > R) {
            L = R = i;
            while (R < s.size() && s[R] == s[R - L]) {
                ++ R;
            }
            zarr[i] = R - L;
            // ensure that R denote the right close interval
            -- R;
        }else {
            int distance_from_begin = i - L;
            if (zarr[distance_from_begin] < R - i + 1) {
                zarr[i] = zarr[distance_from_begin];
            }else {
                cout << "i = " << i << endl;
                L = i;
                while (R < s.size() && s[R] == s[R - L]) {
                    ++ R;
                }
                zarr[i] = R - L;
                -- R;
            }
        }
    }

    return zarr;
}

// assume '$' not in str or ptn
int z_algorithm(const string& str, const string& ptn) {
    string s = ptn + "$" + str;
    cout << "s: " << s << endl;

    vector<int> zarr = get_zarray(s);
    cout << "zarr: ";
    for (auto n : zarr) cout << n << ",";
    cout << endl;

    for (int i = 0; i < zarr.size(); ++ i) {
        if (zarr[i] == ptn.size()) {
            return i - ptn.size();
        }
    }

    return -1;
}


int main() {
    const string str = "BBC ABCDAB ABCDABCDABDE";
    const string ptn = "ABCDABD";
    // const string ptn = "ABAB";
    // const string ptn = "ABCDAXD";
    int ret = z_algorithm(str, ptn);
    if (ret == -1) {
        cout << "\"" << ptn << "\"" << " is not in " << "\"" << str << "\"" << endl;
    }else {
        // cout << "\"" << ptn << "\"" << " is in " << "\"" << str << "\"" << endl;
        cout << "Found " << "\"" << ptn << "\"" << " in " << ret << " of " << "\"" << str << "\"" << endl;
    }

    return 0;
}
