
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])

AgentInfoRnn = namedarraytuple("AgentInfoRnn", ["dist_info", "dist_ext_info", "dist_int_info", "value", "value_int", "prev_rnn_state"])

IcmInfo = namedarraytuple("IcmInfo", [])
NdigoInfo = namedarraytuple("NdigoInfo", ["prev_gru_state"])
RndInfo = namedarraytuple("RndInfo", [])
