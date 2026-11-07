def min_path_sum(grid):
    n = len(grid)
    m = len(grid[0])
    dp = [[0]*m for _ in range(n)]
    dp[0][0] = grid[0][0]

    for i in range(1, n):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, m):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]

    return dp[n-1][m-1]

# 自己写解析函数，不用正则
def parse_matrix(s):
    s = s.strip()
    # 去掉最外层 [[ ]]
    if s.startswith("[[") and s.endswith("]]"):
        s = s[2:-2]
    # 现在形如 "1, 2, 3], [4, 5, 6"
    rows = s.split("], [")
    matrix = []
    for row in rows:
        nums = [int(x.strip()) for x in row.split(",") if x.strip()]
        matrix.append(nums)
    return matrix

line = input().strip()
grid = parse_matrix(line)
print(min_path_sum(grid))

