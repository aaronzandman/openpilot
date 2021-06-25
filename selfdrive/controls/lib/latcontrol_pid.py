import math

from selfdrive.controls.lib.pid import PIController
from selfdrive.controls.lib.drive_helpers import get_steer_max
from common.numpy_fast import clip
from cereal import log
from torque_model.models.fifth_model import predict as model_predict


class LatControlPID():
  def __init__(self, CP):
    self.pid = PIController((CP.lateralTuning.pid.kpBP, CP.lateralTuning.pid.kpV),
                            (CP.lateralTuning.pid.kiBP, CP.lateralTuning.pid.kiV),
                            k_f=CP.lateralTuning.pid.kf, pos_limit=1.0, neg_limit=-1.0,
                            sat_limit=CP.steerLimitTimer)

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, CP, VM, params, lat_plan):
    pid_log = log.ControlsState.LateralPIDState.new_message()
    pid_log.steeringAngleDeg = float(CS.steeringAngleDeg)
    pid_log.steeringRateDeg = float(CS.steeringRateDeg)

    angle_steers_des_no_offset = math.degrees(VM.get_steer_from_curvature(-lat_plan.curvature, CS.vEgo))
    angle_steers_des = angle_steers_des_no_offset + params.angleOffsetDeg

    if CS.vEgo < 0.3 or not active:
      output_steer = 0.0
      pid_log.active = False
      self.pid.reset()
    else:
      steers_max = get_steer_max(CP, CS.vEgo)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max

      # TODO: feedforward something based on lat_plan.rateSteers
      steer_feedforward = angle_steers_des_no_offset  # offset does not contribute to resistive torque
      steer_feedforward *= CS.vEgo**2  # proportional to realigning tire momentum (~ lateral accel)

      deadzone = 0.0

      check_saturation = (CS.vEgo > 10) and not CS.steeringRateLimited and not CS.steeringPressed
      # output_steer = self.pid.update(angle_steers_des, CS.steeringAngleDeg, check_saturation=check_saturation, override=CS.steeringPressed,
      #                                feedforward=steer_feedforward, speed=CS.vEgo, deadzone=deadzone)

      rate_des = 0  # lat_plan.steeringRateDeg if self.op_params.get('model_use_des_rate') else 0
      rate = 0  # CS.steeringRateDeg if self.op_params.get('model_use_rate') else 0
      model_input = [angle_steers_des, CS.steeringAngleDeg, rate_des, rate, CS.vEgo]

      output_steer = model_predict(model_input)[0]
      output_steer = float(clip(output_steer, -1, 1))

      pid_log.active = True
      pid_log.p = self.pid.p
      pid_log.i = self.pid.i
      pid_log.f = self.pid.f
      pid_log.output = output_steer
      pid_log.saturated = bool(self.pid.saturated)

    return output_steer, 0, pid_log
