#include <iostream>
#include <vector>

using namespace std;


// range sum segment tree
class Segment_tree {
    int N;
    vector<int> sum;
    vector<int> nums;
public:
    Segment_tree(vector<int>& arr) {
        N = arr.size();
        sum.resize(4*N + 5);
        nums.resize(N);
        for (auto i = 0; i < arr.size(); ++ i) {
            nums[i] = arr[i];
        }

        build_tree(1, 0, N - 1);
    }
    
    void push_up(int root) {
        sum[root] = sum[root<<1] + sum[root<<1|1];
    }

    void build_tree(int root, int left, int right) {
        if (left == right) {
            sum[root] = nums[left];
            return ;
        }

        int mid = (left + right) >> 1;
        build_tree(root<<1, left, mid);
        build_tree(root<<1|1, mid + 1, right);
        push_up(root);
    }
    
    void update_core(int pos, int val, int left, int right, int root) {
        if (left == right) {
            sum[root] += val;
            return;
        }

        int mid = (left + right) >> 1;
        if (pos <= mid) {
            update_core(pos, val, left, mid, root<<1);
        }else {
            update_core(pos, val, mid+1, right, root<<1|1);
        }
        push_up(root);
    }

    void update(int pos, int val) {
        update_core(pos, val, 0, N-1, 1);
    }

    int query_core(int L, int R, int left, int right, int root) {
        if (L <= left && right <= R) {
            return sum[root];
        }

        int mid = (left + right) >> 1;
        int ans = 0;
        if (L <= mid) {
            ans += query_core(L, R, left, mid, root<<1);
        }
        if (mid < R) {
            ans += query_core(L, R, mid+1, right, root<<1|1);
        }
        return ans;
    }

    int query(int L, int R) {
        return query_core(L - 1, R - 1, 0, N-1, 1);
    }

    void print_tree() {
        for (auto &n : sum) {
            cout << n << ",";
        }
        cout << endl;
    }
};

int main() {
    vector<int> nums = {1,2,3,4,5,6,7,8,9,10};
    Segment_tree tree(nums);

    cout << "original array: " << endl;
    tree.print_tree();

    while (true) {
        cout << "Input 1 for range sum query, 2 for position value update" << endl;
        int opt;
        cin >> opt;
        // range sum query
        if (opt == 1) {
            cout << "please input query range: (l,r)" << endl;
            int l, r;
            cin >> l >> r;
            cout << "range sum is: " << tree.query(l, r) << endl;;
        }
        // position update
        else {
            cout << "pleas input update position and value: (pos, val)" << endl;
            int pos, val;
            cin >> pos >> val;

            tree.update(pos, val);
            cout << "after update, the array is: " << endl;
            tree.print_tree();
        }
    }

    return 0;
}
