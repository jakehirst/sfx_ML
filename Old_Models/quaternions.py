import numpy as np
import quaternionic
import math as m

def height_phi_theta_df_to_quaternions(df):
    quats = quaternionic.array.from_spherical_coordinates((df['phi']*m.pi / 180).tolist(), (df['theta']*m.pi / 180).tolist())
    # i = 0
    # for quat in quats:
    #     print(quat.to_spherical_coordinates * 180 / m.pi)
    #     print(str(df.iloc[i][['phi','theta']].tolist()) + "\n")
    #     i +=1 
    df['quats'] = quats.tolist()
    return df

def quaternions_back_to_sphereical(quat_predictions):
    phi_and_theta_predictions = quaternionic.array(quat_predictions).to_spherical_coordinates * 180 / m.pi
    phi_and_theta_positive_predictions = phi_and_theta_predictions.copy()
    for i in range(len(phi_and_theta_predictions)):
        prediction = phi_and_theta_predictions[i]
        if(prediction[1] < 0): #getting rid of all negative values for theta, and changing them back to positive coordinate system
            phi_and_theta_positive_predictions[i] = [prediction[0], 360 + prediction[1]]
    return phi_and_theta_positive_predictions