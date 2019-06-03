#include <iostream>
#include <string>
#include <vector>

using namespace std;

int main() {
    // int states[2][4] = {1, 0, 2, 0};
    vector<vector<int>> states(2, vector<int>(4,0));
    states[0] = {1, 0, 2, 0};
    for (int i = 0; i < 2; ++ i) {
        for (int j = 0; j < 4; ++ j) {
            cout << states[i][j] << ",";
        }
        cout << endl;
    }
                                                    
    return 0;
}
