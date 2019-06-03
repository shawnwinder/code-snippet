#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;


// weighted, quick-union UF with path compression
class UF {
    int count;
    vector<int> parent;
    vector<int> size;  // weighted nodes
public:
    UF(int N) {
        count = N;
        parent.resize(N);
        size.resize(N);
        for (int i = 0; i < N; ++ i) {
            parent[i] = i;
            size[i] = 1;
        }
    }

    int find(int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];  // path compression
            i = parent[i];
        }
        return i;
    }

    void unions(int a, int b) {
        int pa = find(a);
        int pb = find(b);

        // same parent, no need to union
        if (pa == pb) {
            return;
        }

        // union two sub-tree
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

    void print_parent() {
        for (int i = 0; i < parent.size(); ++ i) {
            cout << parent[i] << ",";
        }
        cout << endl;
    }
};


int main() {
    int N = 10;
    vector<pair<int,int>> relation = {{4,3}, {3,8}, {6,5}, {9,4}, {2,1}, 
        {8,9}, {5,0}, {7,2}, {6,1}, {1,0}, {6,7}};

    UF uf(N); 
    for (const auto &p : relation) {
        uf.unions(p.first, p.second);
        // uf.print_parent();
    }

    cout << "The friend group number is: " << uf.components() << endl;

    return 0;
}
