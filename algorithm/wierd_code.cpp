#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <utility>

using namespace std;

int lengthOfLongestSubstring(string s) {
    const int MAX = 300;
    vector<int> table(256, MAX);
    int cnt = 0;
    int max_length = -1;
    for (int i = 0; i < s.size(); ++ i) {
        if (table[s[i]] == MAX) {
            table[s[i]] = i;
            ++ cnt;
        }else {
            cout << "table[s[i]]: " << table[s[i]] << "," << s[i] << endl;
            max_length = max(max_length, cnt);
            for (int k = 0; k < 256; ++ k) {
                if (table[k] != MAX && table[k] <= table[s[i]]) {
                    cout << "table[k]: " << table[k] << ",";
                    cout << "table[s[i]]: " << table[s[i]] << endl;
                    table[k] = MAX;
                    -- cnt;
                }
                /*
                if (table[k] != -1 && table[k] <= table[s[i]]) {
                    cout << "table[k]: " << table[k] << endl;
                    
                }*/
            }
            table[s[i]] = i;
        }
        
        cout << "table: ";
        for (int i = 'a'; i <= 'z'; ++ i) {
            cout << table[i] << ",";
        }
        cout << endl;
    }
    
    return max_length;
}

int main() {
    string s = "dafasdfeawfasdfas";
    cout << lengthOfLongestSubstring(s) << endl;

    return 0;
}




