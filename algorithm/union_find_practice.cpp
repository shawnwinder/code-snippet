#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

class UF {
    int count;
    vector<int> parent;
    vector<int> size;
public:
    UF (int N){
        count = N;
        parent.resize(N);
        size.resize(N);
        for (int i = 0; i < N; ++ i) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    int find(int id) {
        while (parent[id] != id) {
            parent[id] = parent[parent[id]];
            id = parent[id];
        }
        return id;
    }

    void unions(int a, int b) {
        int pa = find(a);
        int pb = find(b);

        if (pa == pb) {
            return;
        }

        if (size[pa] < size[pb]) {
            parent[pa] = pb;
            size[pb] += size[pa];
        }else {
            parent[pb] = pa;
            size[pa] += size[pb];
        }
        -- count;
    }

    int components() {
        return count;
    }
};

int main() {
    // answer : 5
    int N = 14;
    vector<pair<int,int>> relation = {{4,3}, {3,8}, {6,5}, {9,4}, {2,1}, 
        {8,9}, {5,0}, {7,2}, {6,1}, {1,0}, {6,7}, {12, 13}};

    // answer : 2
    // int N = 10;
    // vector<pair<int,int>> relation = {{4,3}, {3,8}, {6,5}, {9,4}, {2,1}, 
    //     {8,9}, {5,0}, {7,2}, {6,1}, {1,0}, {6,7}};

    UF uf(N); 
    for (const auto &p : relation) {
        uf.unions(p.first, p.second);
        // uf.print_parent();
    }

    cout << "The friend group number is: " << uf.components() << endl;

    return 0;
}
