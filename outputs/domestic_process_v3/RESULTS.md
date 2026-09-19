# 国内处理阶段区分：配对演示

6 个情景，各运行连续种子 0–9，共 60 次；每组人口、网络、初始条件及潜在事件随机流配对。参数未校准。
关闭国内落实不妨碍客户渠道解决；关闭客户后的企业响应不妨碍国内路径落实。全部解决关闭时，机构仍可形成支持结果，但案件不会自行解决。
图左为机构待处理案件，中间为已获支持但待落实案件，右为全部未解决案件；每条线表示一种参数情景，阴影为跨种子 ±1 样本标准差。
实施失败后案件仍未解决，但不再属于待落实队列。队列较短不必然意味着解决更多，必须结合右图。
baseline 为默认；implementation_off 关闭落实；implementation_slow 只延长落实等待；implementation_zero_effect 只将落实效果置零；customer_response_off 只关闭客户后的企业响应；all_resolution_off 同时关闭两条路径的最终解决过程。
结果为机制验证，不代表现实机构表现。完整参数、哈希与检查见 verification.json。

                            unresolved  cumulative_resolved  pending_domestic_implementation
scenario                                                                                    
all_resolution_off                27.8                  0.0                             11.4
baseline                          16.7                 15.5                              0.2
customer_response_off             20.5                 10.6                              0.1
implementation_off                23.6                  5.3                             10.3
implementation_slow               16.7                 14.7                              1.0
implementation_zero_effect        23.4                  5.6                              0.1
