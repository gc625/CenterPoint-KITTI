import csv
import numpy as np
import mmcv
from matplotlib import pyplot as plt
from math import sin, cos, pi
import pandas as pd
import os
import argparse

class Lrr30:

    def __init__(self, path, target_sensor):
        self.data = []
        self.spm_point_cloud = {}

        self.read_raw(path)
        self.draw_spmpcd(target_sensor)
        plt.plot(self.timestamps)
        plt.xlabel('frame')
        plt.ylabel('timestamp')
        plt.savefig('timestamps.png',dpi=500)
        
    def read_raw(self, path):
        num_frames = 0
        num_incomp_frames = 0
        with open(path) as f:
            reader = csv.reader(f)
            self.timestamps = []
            for li, line in enumerate(reader):
                # arrgement version
                if line.pop(0) != 'hasco-lrr30-v1':
                    continue
                line_cp = line[:]
                num_frames += 1
                try:
                    frame = self.dict_frame(line)
                    self.timestamps.append(frame['recvTime'])
                except ValueError:
                    # print('{} frame is not compatible'.format(li))
                    num_incomp_frames += 1
                    if 'frame' not in locals():
                        continue
                    frame['left'] = line

                self.data.append(frame)
        print('Done reading raw radar data. '\
            f'{num_incomp_frames} out of {num_frames} frames were incompatible.')

    @staticmethod
    def dict_frame(line):
        frame = {}
        # relative time of Aseva receives radar data
        frame['recvTime'] = float(line.pop(0))

        # header
        frame['header'] = header = {}

        header['iEthProtcolVersionMajor'] = int(line.pop(0))
        header['iEthProtcolVersionMirror'] = int(line.pop(0))
        header['iEthTotalPackets'] = int(line.pop(0))
        header['iPlatformSWversion'] = int(line.pop(0))
        header['iPlatformRFversion'] = int(line.pop(0))
        header['iPlatformHWversion'] = int(line.pop(0))
        header['iSensorTotalCount'] = int(line.pop(0))
        header['isMaster'] = int(line.pop(0))
        header['iSensorIndex'] = int(line.pop(0))
        header['iRes'] = int(line.pop(0))
        header['iCurrentFrameTimeStamp'] = int(line.pop(0))
        header['iCurrentFrameID'] = int(line.pop(0))
        line.pop(0)
        header['iSensorIndex_Customer'] = int(line.pop(0))  # TODO

        header['vehicleInfoShow'] = vehicleInfoShow = {}
        vehicleInfoShow['Velocity'] = float(line.pop(0))
        vehicleInfoShow['SteerAngleVal'] = float(line.pop(0))
        vehicleInfoShow['YawRate'] = float(line.pop(0))
        vehicleInfoShow['TurnRadius'] = float(line.pop(0))
        vehicleInfoShow['GearPosVal'] = int(line.pop(0))

        header['splitAA'] = int(line.pop(0))

        header['diagInfoShow'] = diagInfoShow = {}
        diagInfoShow['EcuFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['McuFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['PmicFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['MmicFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['PowerFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['SwFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['CommFaulty'] = bool(int(line.pop(0)))
        diagInfoShow['MemFaulty'] = bool(int(line.pop(0)))

        # spm
        frame['spm'] = spm = {}

        spm['fTimeStamp'] = float(line.pop(0))

        spm['sVdyOutput'] = sVdyOutput = {}
        sVdyOutput['fTimeStamp'] = float(line.pop(0))
        sVdyOutput['fVelocity'] = float(line.pop(0))
        sVdyOutput['fVelocityStd'] = float(line.pop(0))
        sVdyOutput['bVelocityValid'] = bool(int(line.pop(0)))
        sVdyOutput['fYawRate'] = float(line.pop(0))
        sVdyOutput['fYawRateStd'] = float(line.pop(0))
        sVdyOutput['bYawRateValid'] = bool(int(line.pop(0)))
        sVdyOutput['fLngAccel'] = float(line.pop(0))
        sVdyOutput['fLngAccelStd'] = float(line.pop(0))
        sVdyOutput['bLngAccelValid'] = bool(int(line.pop(0)))
        sVdyOutput['fLatAccel'] = float(line.pop(0))
        sVdyOutput['fLatAccelStd'] = float(line.pop(0))
        sVdyOutput['bLatAccelValid'] = bool(int(line.pop(0)))
        sVdyOutput['fSteeringWheelAngle'] = float(line.pop(0))
        sVdyOutput['bSteeringWheelAngleValid'] = bool(int(line.pop(0)))
        sVdyOutput['fCurvature'] = float(line.pop(0))
        sVdyOutput['fCurvatureStd'] = float(line.pop(0))
        sVdyOutput['bCurvatureValid'] = bool(int(line.pop(0)))
        sVdyOutput['eMotionState'] = int(line.pop(0))

        spm['eWaveForm'] = int(line.pop(0))
        spm['iTargetsCount'] = int(line.pop(0))
        spm['iEnvironment_infor'] = int(line.pop(0))

        spm['targets'] = targets = {}
        for i in range(spm['iTargetsCount']):
            targets[i] = target = {}
            target['fAzangle'] = float(line.pop(0))
            target['fElangle'] = float(line.pop(0))
            target['fRange'] = float(line.pop(0))
            target['fPower'] = float(line.pop(0))
            target['fSpeed'] = float(line.pop(0))
            target['fSNR'] = float(line.pop(0))
            target['fRCS'] = float(line.pop(0))
            target['fProbability'] = float(line.pop(0))
            target['uAmbigState'] = int(line.pop(0))
            target['bPeakFlag'] = int(line.pop(0))
            target['fStdRange'] = float(line.pop(0))
            target['fStdSpeed'] = float(line.pop(0))
            target['fStdAzangle'] = float(line.pop(0))
            target['fStdElangle'] = float(line.pop(0))
            target['uStateFlag'] = int(line.pop(0))
            target['iARID'] = int(line.pop(0))
            target['iAVID'] = int(line.pop(0))
            target['iBRID'] = int(line.pop(0))
            target['iBVID'] = int(line.pop(0))

        # spm['SPM_sRadarPara_t'] = SPM_sRadarPara_t = {}
        # SPM_sRadarPara_t['eAntennaType'] = int(line.pop(0))
        # SPM_sRadarPara_t['fStartFreq'] = float(line.pop(0))
        # SPM_sRadarPara_t['fCenterFreq'] = float(line.pop(0))
        # SPM_sRadarPara_t['fBandWidth'] = float(line.pop(0))
        # SPM_sRadarPara_t['fPeriodTime'] = float(line.pop(0))
        # SPM_sRadarPara_t['fRangeAmbig'] = float(line.pop(0))
        # SPM_sRadarPara_t['fVeloAmbig'] = float(line.pop(0))
        # SPM_sRadarPara_t['fAzAngleAmbig'] = float(line.pop(0))
        # SPM_sRadarPara_t['fElAngleAmbig'] = int(line.pop(0))
        # SPM_sRadarPara_t['fRangeRes'] = int(line.pop(0))
        # SPM_sRadarPara_t['fVeloRes'] = float(line.pop(0))
        # SPM_sRadarPara_t['fAzAngleRes'] = float(line.pop(0))
        # SPM_sRadarPara_t['fElAngleRes'] = float(line.pop(0))
        # SPM_sRadarPara_t['uSensorId'] = int(line.pop(0))

        spm['Ptp_Seconds'] = int(line.pop(0))
        spm['Ptp_NanoSeconds'] = int(line.pop(0))
        #
        # # track
        # frame['track'] = track = {}
        #
        # track['fTimeStamp'] = float(line.pop(0))
        # track['iFaultCode'] = int(line.pop(0))
        #
        # track['sObjectList'] = sObjectList = {}
        # sObjectList['fTimeStamp'] = float(line.pop(0))
        # sObjectList['iObjectsNum'] = int(line.pop(0))

        # sObjectList['sObjects'] = sObjects = {}
        # for i in range(sObjectList['iObjectsNum']):

        return frame

    @staticmethod
    def spm2pcd(speed, range, azangle, elangle):
        """
        convert raw data to pcd
        :param speed:
        :param range:
        :param azangle:
        :param elangle:
        :return:
        """
        x = range * cos((azangle) * pi / 180.) * cos((elangle) * pi / 180.)
        y = range * sin((azangle) * pi / 180.) * cos((elangle) * pi / 180.)
        z = range * sin((elangle) * pi / 180.)
        vx = speed * cos((azangle) * pi / 180.)
        vy = speed * sin((azangle) * pi / 180.)

        return x, y, z, vx, vy

    def draw_spmpcd(self, target_sensor):
        """
        draw spm data with .pcd from raw
        :return:
        """

        target_dict = {
            "front": 0,
            "right": 1,
            "left": 4
        }

        target_idx = target_dict[target_sensor]

        for frame in self.data:
            points = []
            sensor_idx = frame['header']['iSensorIndex']

            if sensor_idx == target_idx:
                targets = frame['spm']['targets']
                for i in range(len(targets)):
                    target = targets[i]
                    x, y, z, vx, vy = self.spm2pcd(target['fSpeed'], target['fRange'], target['fAzangle'], target['fElangle'])
                    point = [x, y, z, vx, vy, target['fPower'], target['fRCS'], target['fSpeed']]
                    points.append(point)

                points = np.array(points)
                self.spm_point_cloud[frame['recvTime']] = points

def read_raw(load_dir, save_dir, base_ts: dict):
    save_dir = os.path.join(save_dir, 'inhouse_format', 'radar_raw')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    lrr = Lrr30(os.path.join(load_dir, 'input', 'raw', 'raw.csv'), target_sensor="front")
    base_ts = base_ts['local']
    num_pcs = 0
    num_pnts = 0
    num_warnings = 0
    num_saves = 0
    for k in mmcv.track_iter_progress(lrr.spm_point_cloud):

        current_ts = int(k * 1e3) + base_ts
        v = lrr.spm_point_cloud[k]

        if np.size(v,0)>0:
            num_pnts += len(v[:,0])
            num_pcs +=1
            dis = np.sqrt(v[:,0]**2+v[:,1]**2+v[:,2]**2)
            if dis.max()<80:
                save_path = os.path.join(save_dir, str(current_ts).zfill(13) + ".csv")
                if num_warnings < 10 and os.path.exists(save_path):
                    print(f'WARN: {save_path} already exists')
                    num_warnings += 1
                data = pd.DataFrame(v)
                data.to_csv(save_path)
                # print("saving radar frame: ", str(k))
                num_saves += 1
            # else: 
            #     raise("Not Single Distance Mode")
    print(f'Saved {num_saves} radar frames.')
    avg_pnts = num_pnts/num_pcs
    print('Average points per scan is {}'.format(avg_pnts))

def main():
    parser = argparse.ArgumentParser(description='Read raw radar sweeps')
    parser.add_argument('data_root', help='path to root of directory containing unprocessed data')
    args = parser.parse_args()
    read_raw(args.data_root)
    

if __name__ == '__main__':
    main()
