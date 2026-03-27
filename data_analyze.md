============================================================
1. 详细题型分布
============================================================
type
bit_manipulation      1602
physics               1597
unit_conversion       1594
encryption            1576
numeral_conversion    1576
mathematics           1555
Name: count, dtype: int64

总计: 9500 条

============================================================
2. 各题型提示词长度统计
============================================================
                     count        mean        std  ...    50%    75%    max
type                                               ...                     
bit_manipulation    1602.0  478.565543  23.321178  ...  489.0  489.0  510.0
encryption          1576.0  370.078680  49.663581  ...  370.0  412.0  485.0
mathematics         1555.0  195.138264   9.782291  ...  195.0  205.0  212.0
numeral_conversion  1576.0  219.128173   9.586038  ...  219.0  228.0  241.0
physics             1597.0  320.261741  27.559853  ...  319.0  352.0  357.0
unit_conversion     1594.0  222.240276  17.824614  ...  223.0  244.0  246.0

[6 rows x 8 columns]

============================================================
2. 各题型答案长度统计
============================================================
                     count       mean       std   min   25%   50%   75%   max
type                                                                         
bit_manipulation    1602.0   8.000000  0.000000   8.0   8.0   8.0   8.0   8.0
encryption          1576.0  25.453046  4.732497  13.0  22.0  25.0  29.0  39.0
mathematics         1555.0   2.943408  0.923391   1.0   2.0   3.0   4.0   4.0
numeral_conversion  1576.0   4.028553  1.508435   1.0   3.0   4.0   5.0   8.0
physics             1597.0   5.047589  0.614356   3.0   5.0   5.0   5.0   6.0
unit_conversion     1594.0   4.908407  0.288542   4.0   5.0   5.0   5.0   5.0

============================================================
3. 题型-答案格式交叉分布表
============================================================
answer_format       binary  decimal  numeric  roman  text
type                                                     
bit_manipulation      1602        0        0      0     0
encryption               0        0        0      0  1576
mathematics             32        0      607      0   916
numeral_conversion       0        0        0   1576     0
physics                  0     1597        0      0     0
unit_conversion          0     1594        0      0     0

============================================================
4. 常见提示词开头
============================================================
[A] 1602条: In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary nu...
[B] 1597条: In Alice's Wonderland, the gravitational constant has been secretly changed. Her...
[C] 1594条: In Alice's Wonderland, a secret unit conversion is applied to measurements. For ...
[D] 1576条: In Alice's Wonderland, secret encryption rules are used on text. Here are some e...
[E] 1576条: In Alice's Wonderland, numbers are secretly converted into a different numeral s...
[F] 1555条: In Alice's Wonderland, a secret set of transformation rules is applied to equati...

============================================================
4. 常见提示词开头-题型交叉分布表
============================================================
type   bit_manipulation  encryption  ...  physics  unit_conversion
label                                ...                          
A                  1602           0  ...        0                0
B                     0           0  ...     1597                0
C                     0           0  ...        0             1594
D                     0        1576  ...        0                0
E                     0           0  ...        0                0
F                     0           0  ...        0                0

[6 rows x 6 columns]
