from pdm.utils.definitions import filter

# If both values are present in the filter, It will filter on ge on the first values and le on the second value.
# only first value present and it will filter on ge of that given value
# only 2nd value present and it will filter on le of that given value
prep_moter_filter = [
    [filter(tag="motor_effect", value=90), None],
    [filter(tag="kv_flow", value=8900), filter(tag="motor_effect", value=200)],
    [filter(tag="kv_flow", value=7500), None],
]
prep_rpm_filter = [
    [filter(tag="kv_flow", value=7500), None],
    [filter(tag="kv_flow", value=9700), filter(tag="pump_rotation", value=200)],
]
prep_rpm_vs_motor_filter = [
    [filter(tag="motor_effect", value=0), None],
    [filter(tag="pump_rotation", value=130), None],
]
