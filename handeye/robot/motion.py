"""UR motion capability, separate from calibration state collection."""


def connect_rtde(host, *, receive_only=False):
    import rtde_receive
    receiver = rtde_receive.RTDEReceiveInterface(host)
    if receive_only:
        return None, receiver
    try:
        import rtde_control
        return rtde_control.RTDEControlInterface(host), receiver
    except Exception:
        receiver.disconnect()
        raise
