#include <iostream>
#include <queue>
#include <algorithm>

using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int n) : val(n), left(nullptr), right(nullptr) {}
};

TreeNode* create_tree(const vector<int>& nums) {
    if (nums.empty()) {
        return nullptr;
    }
    if (nums[0] == -1) {
        return nullptr;
    }

    queue<TreeNode*> q;
    TreeNode* root = new TreeNode(nums.front());
    q.push(root);
    int i = 0;
    while (!q.empty()){
        auto cur = q.front();
        q.pop();
        // left
        if (i < nums.size()) {
            if (nums[i] != -1) {
                cur->left = new TreeNode(nums[i]);
                q.push(cur->left);
            }else {
                cur->left = nullptr;
            }
            ++ i;
        }else {
            cur->left = nullptr;
        }
        // right
        if (i < nums.size()) {
            if (nums[i] != -1) {
                cur->right = new TreeNode(nums[i]);
                q.push(cur->right);
            }else {
                cur->right= nullptr;
            }
            ++ i;
        }else {
            cur->right= nullptr;
        }
    }

    return root;
}

bool is_complete_binary_tree(TreeNode* root) {
    if (!root) {
        return false;
    }
    queue<TreeNode*> q;
    q.push(root);
    while (q.front() != nullptr) {
        auto cur = q.front();
        q.pop();
        q.push(cur->left);
        q.push(cur->right);
    }
    
    while (!q.empty()) {
        if (q.front() != nullptr) {
            return false;
        }
        q.pop();
    }

    return true;
}

int main() {
    // vector<int> nums = {1,2,3,4,5,6};
    vector<int> nums = {1,2,3,4,5,6,-1,7};
    TreeNode* root = create_tree(nums);

    if (is_complete_binary_tree(root)) {
        cout << "yes" << endl;
    }else {
        cout << "no" << endl;
    }

    return 0;
}
