#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <utility>
#include <algorithm>

using namespace std;

void path_search(vector<vector<int>>& data, vector<vector<int>>& mp,
        vector<vector<bool>>& visited, int row, int col, 
        int path_money, int& max_money) {
    vector<vector<int>> dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    max_money = max(max_money, path_money);
    int n = data.size(), m = data[0].size();
    for (auto dir : dirs) {
        int x = row + dir[0];
        int y = col + dir[1];
        if (x >= 0 && x < n && y >= 0 && y < m && !visited[x][y] && 
                data[x][y] < data[row][col] && path_money + 1 > mp[row][col]) {
            visited[x][y] = true;
            path_search(data, mp, visited, x, y, path_money + 1, max_money);

            visited[x][y] = false;
            path_money -= 1;
        }
    }
}

int way_to_rich(vector<vector<int>>& data) {
    // edge cases
    if (data.empty()) {
        return 0;
    }

    // general cases
    int n = data.size(), m = data[0].size();
    vector<vector<int>> mp(n, vector<int>(m, 0));
    int richest = -1;
    for (int i = 0; i < n; ++ i) {
        for (int j = 0; j < m; ++ j) {
            vector<vector<bool>> visited(n, vector<bool>(m, false));
            visited[i][j] = true;
            int pos_money = 0;
            path_search(data, mp, visited, i, j, 1, pos_money);
            mp[i][j] = pos_money;
            richest = pos_money> richest ? pos_money : richest;
        }
    }

    return richest;
}

int main() {
    int n = 6, m = 6;
    vector<vector<int>> data = {
        {32, 34, 7, 33, 21, 2 },
        {13, 12, 3, 11, 26, 36},
        {16, 30, 22, 1, 24, 14},
        {20, 23, 25, 5, 19, 29},
        {27, 15, 9, 17, 31, 4},
        {6, 18, 8, 10, 35, 28}
    };

    int max_money = way_to_rich(data);

    cout << "max money is: " << max_money << endl;

    return 0;
}
