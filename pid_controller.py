import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.dt = dt
        self.integral = 0
        self.prev_error = 0

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint
        
    def update(self, measured_value):
        error = self.setpoint - measured_value
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output