#!/usr/bin/env python
# coding: utf-8

# # **Christian Gilson - Tracking Data Assignment**
#
# Sunday 11th October 2020
#
# ---

# In[1]:


import pandas as pd
import numpy as np
import datetime

# imports required by data prep functions
import json

# Laurie's libraries
import scipy.signal as signal
import matplotlib.animation as animation

# removing annoying matplotlib warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import re
import os
from collections import Counter, defaultdict


# plotting
import matplotlib.pyplot as plt

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500

signalityRepo = r'2019/Tracking Data/'
movieRepo = r'Movies/'


# # **1)** Data Preparation Functions

# In[2]:


def initialise_dic_tracks(df_homePlayers, df_awayPlayers):
    """
    Initialises dictionaries for both home and away player locations
    """

    dic_home_tracks = {}
    dic_away_tracks = {}

    for homePlayer in df_homePlayers.playerIndex:
        for xy in ['x','y']:
            dic_home_tracks[f'Home_{homePlayer}_{xy}'] = []

    for awayPlayer in df_awayPlayers.playerIndex:
        for xy in ['x','y']:
            dic_away_tracks[f'Away_{awayPlayer}_{xy}'] = []

    return dic_home_tracks, dic_away_tracks


# In[3]:


def populate_df_tracks(homeAway, homeAway_tracks, playersJerseyMapping, dic_tracks, df_players):
    """
    For a given team (home OR away), will transform the JSON track data to produce a dataframe just like Laurie's
    """

    lst_playerJerseys = df_players.jersey_number.values

    # iterating through frames for home/away team
    for n, frame in enumerate(homeAway_tracks):

        lst_playerJerseysPerFrame = []

        for player in frame:
            jersey_number = player.get('jersey_number')
            playerIndex = playersJerseyMapping[jersey_number]
            x,y = player.get('position', [np.nan, np.nan])

            # keeping track of jerseys that have a position for that frame
            lst_playerJerseysPerFrame.append(jersey_number)

            dic_tracks[f'{homeAway}_{playerIndex}_x'].append(x)
            # flipping the y axis to make the data sync with Laurie's plotting methods
            dic_tracks[f'{homeAway}_{playerIndex}_y'].append(-1*y)

        # list of jerseys that aren't in the frame
        lst_playerJerseysNotInFrame = list(set(lst_playerJerseys) - set(lst_playerJerseysPerFrame))

        # adding the jerseys that aren't in frame and providing an x,y position of nan, nan
        for jersey_number in lst_playerJerseysNotInFrame:
            playerIndex = playersJerseyMapping[jersey_number]
            x,y = [np.nan, np.nan]

            dic_tracks[f'{homeAway}_{playerIndex}_x'].append(x)
            dic_tracks[f'{homeAway}_{playerIndex}_y'].append(y)

    # transforming tracking dic to a tracking dataframe
    df_tracks = pd.DataFrame(dic_tracks)

    return df_tracks


# In[4]:


def to_single_playing_direction(home,away):
    """
    Switches x and y co-ords with negative sign in the second half
    Requires the co-ords to be symmetric about 0,0 (i.e. going from roughly -60 to +60 in the x direction and -34 to +34 in the y direction)
    """
    for team in [home,away]:
        second_half_idx = team.Period.idxmax(2)
        columns = [c for c in team.columns if c[-1].lower() in ['x','y']]
        team.loc[second_half_idx:,columns] *= -1
    return home,away


# In[5]:


def shoot_direction(gk_x_position):
    """
    Produces either 1 (L2R) or -1 (R2L) based on GK position
    """
    if gk_x_position > 0:
        # shooting right-to-left
        return -1
    else:
        # shotting left-to-right
        return 1


# In[6]:


def parse_raw_to_df(signalityRepo, rootFileName, interpolate=True):
    """
    Takes raw root of a match string e.g. 20190930.Hammarby-Örebrö and transforms it into 4 dataframes:
    1) home players
    2) away players
    3) home tracking
    4) away tracking
    """

    lst_df_home = []
    lst_df_away = []

    for half in ['.1','.2']:

        # producing filename prefix (just need to add either "-info_live.json" or "-tracks.json")
        fileNamePrefix = rootFileName + half

        # load info
        ## looks like the info JSON is duplicated between the two halves
        with open(os.path.join(signalityRepo, f'{fileNamePrefix}-info_live.json')) as f:
            info = json.load(f)

        # load tracks
        with open(os.path.join(signalityRepo, f'{fileNamePrefix}-tracks.json')) as f:
            tracks = json.load(f)

        # unpacking info
        ## looks like .1 and .2 files are duplicated, so just looking at the .1 (first half file)
        if half == '.1':
            matchId = info.get('id')
            venueId = info.get('venueId')
            timeStart = info.get('time_start')
            pitchLength, pitchWidth = info.get('calibration').get('pitch_size')
            homeTeam = info.get('team_home_name')
            awayTeam = info.get('team_away_name')

            # unpacking players
            homePlayers = info.get('team_home_players')
            awayPlayers = info.get('team_away_players')
            homeLineup = info.get('team_home_lineup')
            awayLineup = info.get('team_away_lineup')
            homeLineupSwitch = {homeLineup[i]:i for i in homeLineup}
            awayLineupSwitch = {awayLineup[i]:i for i in awayLineup}

            # putting player metadata in dataframe
            df_homePlayers = pd.DataFrame(homePlayers)
            df_awayPlayers = pd.DataFrame(awayPlayers)
            df_homePlayers['teamName'] = homeTeam
            df_awayPlayers['teamName'] = awayTeam

            # adding matchId to the player dataframes
            df_homePlayers['matchId'] = matchId
            df_awayPlayers['matchId'] = matchId
            df_homePlayers['matchName'] = rootFileName
            df_awayPlayers['matchName'] = rootFileName

            # adding 1-11 + sub player indices (will probably use these for the final column names like Laurie in the tracks df)
            df_homePlayers['playerIndex'] = [int(homeLineupSwitch[i]) if i in homeLineupSwitch else np.nan for i in df_homePlayers.jersey_number.values]
            df_awayPlayers['playerIndex'] = [int(awayLineupSwitch[i]) if i in awayLineupSwitch else np.nan for i in df_awayPlayers.jersey_number.values]
            df_homePlayers.loc[pd.isna(df_homePlayers['playerIndex']) == True, 'playerIndex'] = np.arange(int(np.nanmax(df_homePlayers.playerIndex))+1, len(df_homePlayers)+1)
            df_awayPlayers.loc[pd.isna(df_awayPlayers['playerIndex']) == True, 'playerIndex'] = np.arange(int(np.nanmax(df_awayPlayers.playerIndex))+1, len(df_awayPlayers)+1)
            df_homePlayers['playerIndex'] = df_homePlayers.playerIndex.apply(lambda x: int(x))
            df_awayPlayers['playerIndex'] = df_awayPlayers.playerIndex.apply(lambda x: int(x))

            # re-jigging cols and re-ordering rows
            df_homePlayers = df_homePlayers[['matchId','matchName','teamName','playerIndex','jersey_number','name']].sort_values('playerIndex')
            df_awayPlayers = df_awayPlayers[['matchId','matchName','teamName','playerIndex','jersey_number','name']].sort_values('playerIndex')

            homePlayersJerseyMapping = {i:j for i, j in zip(df_homePlayers.jersey_number, df_homePlayers.playerIndex)}
            awayPlayersJerseyMapping = {i:j for i, j in zip(df_awayPlayers.jersey_number, df_awayPlayers.playerIndex)}

        ## parsing the track data
        phase = int(half[-1])

        # extracting home and away tracks
        home_tracks = [i.get('home_team') for i in tracks]
        away_tracks = [i.get('away_team') for i in tracks]

        # ball tracks
        ball_tracks = [i.get('ball') for i in tracks]
        ball_tracks_position = [(i.get('position')[0],i.get('position')[1],i.get('position')[2]) if i.get('position') != None else (np.nan, np.nan, np.nan) for i in ball_tracks]
        ball_x = [i[0] for i in ball_tracks_position]
        # flipping the y-coordinate
        ball_y = [-1*i[1] for i in ball_tracks_position]
        ball_z = [i[2] for i in ball_tracks_position]
        ball_jerseyPossession = [i.get('player') for i in ball_tracks]
        ball_jerseyPossession = [int(i) if pd.isna(i) == False else np.nan for i in ball_jerseyPossession]

        # match timestamps
        match_time = [i.get('match_time') for i in tracks]
        period = [i.get('phase') for i in tracks]
        timeStamp = pd.to_datetime([datetime.datetime.utcfromtimestamp(i.get('utc_time')/1000) for i in tracks])

        # unpacking tracks
        ## 1) initialising dictionaries for home and away teams
        dic_home_tracks, dic_away_tracks = initialise_dic_tracks(df_homePlayers, df_awayPlayers)
        ## 2) producing home tracking dataframe
        df_home_tracks = populate_df_tracks('Home', home_tracks, homePlayersJerseyMapping, dic_home_tracks, df_homePlayers)
        ## 3) producing away tracking dataframe
        df_away_tracks = populate_df_tracks('Away', away_tracks, awayPlayersJerseyMapping, dic_away_tracks, df_awayPlayers)

        # putting things together
        ## 1) home
        df_home_tracks['ball_x'] = ball_x
        df_home_tracks['ball_y'] = ball_y
        df_home_tracks['ball_z'] = ball_z

        # at this point we just have player and ball positions in the dataframe, so providing option now to interpolate
        # linearly interpolating (inside only) when there are enclosed NaNs - this is the shortest path, and will thus be the slowest in a set amount of time, so won't overestimate speed / acceleration when we're missing data
        if interpolate:
            df_home_tracks = df_home_tracks.interpolate(method='linear', limit_area='inside')

        # and now adding things where we wouldn't want there to be any interpolation (like the ball_jerseyPossession)
        df_home_tracks['halfIndex'] = df_home_tracks.index
        df_home_tracks['matchId'] = matchId
        df_home_tracks['matchName'] = rootFileName
        df_home_tracks['Period'] = period
        df_home_tracks['Time [s]'] = np.array(match_time) / 1000
        df_home_tracks['TimeStamp'] = timeStamp
        df_home_tracks['ball_jerseyPossession'] = ball_jerseyPossession

        ## 2) away
        df_away_tracks['ball_x'] = ball_x
        df_away_tracks['ball_y'] = ball_y
        df_away_tracks['ball_z'] = ball_z

        # option to interpolate, like with the home team
        if interpolate:
            df_away_tracks = df_away_tracks.interpolate(method='linear', limit_area='inside')

        df_away_tracks['halfIndex'] = df_away_tracks.index
        df_away_tracks['matchId'] = matchId
        df_away_tracks['matchName'] = rootFileName
        df_away_tracks['Period'] = period
        df_away_tracks['Time [s]'] = np.array(match_time) / 1000
        df_away_tracks['TimeStamp'] = timeStamp
        df_away_tracks['ball_jerseyPossession'] = ball_jerseyPossession

        lst_df_home.append(df_home_tracks)
        lst_df_away.append(df_away_tracks)

    # combining the first and second half data
    df_homeTracks = pd.concat(lst_df_home, ignore_index=True)
    df_awayTracks = pd.concat(lst_df_away, ignore_index=True)

    # getting a master index for the full game
    df_homeTracks['index'] = df_homeTracks.index
    df_awayTracks['index'] = df_awayTracks.index

    # forcing the second half of the match to follow the same direction as the first
    df_homeTracks, df_awayTracks = to_single_playing_direction(df_homeTracks, df_awayTracks)

    # use GK x position to know whether team is shooting right-to-left or left-to-right.
    avHomeGKxTrack = df_homeTracks.Home_1_x.mean()
    avAwayGKxTrack = df_awayTracks.Away_1_x.mean()

    # apply shooting direction to both home and away dataframes
    df_homeTracks['shootingDirection'] = shoot_direction(avHomeGKxTrack)
    df_awayTracks['shootingDirection'] = shoot_direction(avAwayGKxTrack)

    return df_homePlayers, df_awayPlayers, df_homeTracks, df_awayTracks, pitchLength, pitchWidth, homePlayersJerseyMapping, awayPlayersJerseyMapping


# # **2)** Calculate first and second derivatives of position: velocity and acceleration

# In[7]:


def calc_opp_goal_position(shootingDirection, pitchLength):
    """
    Outputs either +1 or -1 if team shooting left-to-right or right-to-left, respectively.
    """

    # 1 = left-to-right
    if shootingDirection == 1:
        return (pitchLength/2, 0)
    # -1 = right-to-left
    else:
        return (-1*pitchLength/2, 0)


# In[8]:


def calc_player_velocities(team, pitchLength = 105, smoothing = True, filter_ = 'Savitzky-Golay', window = 7, polyorder = 1, maxspeed = 12):
    """
    Calculate player x,y components of velocities and acceleration
    Also calculates scalar quantities for velocity (i.e. speed), acceleration, and player distance to goal

    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is Savitzky-Golay, which fits a polynomial of order 'polyorder' to the data within each window
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velocity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN.

    """
    # remove any velocity data already in the dataframe
    team = remove_player_velocity_acceleration_distance(team)

    # extract the shooting direction (+1 L2R, -1 R2L)
    shootingDirection = team.shootingDirection.values[0]
    # getting the opposite goal position
    goal_x, goal_y = calc_opp_goal_position(shootingDirection, pitchLength)

    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in team.columns if c[:4] in ['Home','Away'] ] )

    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = team['Time [s]'].diff()

    # index of first frame in second half
    second_half_idx = team.Period.idxmax(2)

    # estimate velocities for players in team
    # cycle through players individually
    for player in player_ids:

        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = team[f'{player}_x'].diff() / dt
        vy = team[f'{player}_y'].diff() / dt

        # calculating distance to goal
        # dy will always just be the y position as goal_y is always 0 by definition using current co-ord system, but leaving in this redundancy for now
        # just incase the co-ord system changes for different applications
        dx = team[f'{player}_x'] - goal_x
        dy = team[f'{player}_y'] - goal_y
        D = np.sqrt(dx**2 + dy**2)

        if maxspeed > 0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt(vx**2 + vy**2)
            vx[raw_speed>maxspeed] = np.nan
            vy[raw_speed>maxspeed] = np.nan

        if smoothing:
            if filter_=='Savitzky-Golay':

                # calculate first half velocity
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)

            elif filter_=='moving average':

                ma_window = np.ones( window ) / window
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' )
                vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' )
                vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' )

        # acceleration components: second derivative of position
        ax = vx.diff() / dt
        ay = vy.diff() / dt

        # acceleration smoothing
        if smoothing:
            if filter_=='Savitzky-Golay':

                # calculate first half acceleration
                ax.loc[:second_half_idx] = signal.savgol_filter(ax.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                ay.loc[:second_half_idx] = signal.savgol_filter(ay.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                # calculate second half acceleration
                ax.loc[second_half_idx:] = signal.savgol_filter(ax.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                ay.loc[second_half_idx:] = signal.savgol_filter(ay.loc[second_half_idx:],window_length=window,polyorder=polyorder)

            elif filter_=='moving average':

                ma_window = np.ones( window ) / window
                # calculate first half acceleration
                ax.loc[:second_half_idx] = np.convolve( ax.loc[:second_half_idx] , ma_window, mode='same' )
                ay.loc[:second_half_idx] = np.convolve( ay.loc[:second_half_idx] , ma_window, mode='same' )
                # calculate second half acceleration
                ax.loc[second_half_idx:] = np.convolve( ax.loc[second_half_idx:] , ma_window, mode='same' )
                ay.loc[second_half_idx:] = np.convolve( ay.loc[second_half_idx:] , ma_window, mode='same' )


        # put player speed in x,y direction, and total speed back in the data frame
        team[f'{player}_vx'] = vx
        team[f'{player}_vy'] = vy
        team[f'{player}_speed'] = np.sqrt( vx**2 + vy**2 )


        team[f'{player}_ax'] = ax
        team[f'{player}_ay'] = ay
        team[f'{player}_acceleration'] = np.sqrt( ax**2 + ay**2 )

        team[f'{player}_D'] = D

    return team


def remove_player_velocity_acceleration_distance(team):
    """
    Clean up function: removes velocities, acceleration, and distance to goal
    """
    # remove player velocities, acceleration, and distance to goal measures that are already in the 'team' dataframe
    # so that they can be cleanly re-calculated
    columns = [c for c in team.columns if c.split('_')[-1] in ['vx','vy','ax','ay','speed','acceleration','D']]
    team = team.drop(columns=columns)
    return team


# # **3)** Functions to create plots and videos

# In[9]:


def plot_pitch( field_dimen = (106.0,68.0), field_color ='green', linewidth=2, markersize=20):
    """ plot_pitch

    Plots a soccer pitch. All distance units converted to meters.

    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20

    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    fig,ax = plt.subplots(figsize=(12,8)) # create a figure
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color=='green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke' # line color
        pc = 'w' # 'spot' colors
    elif field_color=='white':
        lc = 'k'
        pc = 'k'
    # ALL DIMENSIONS IN m
    border_dimen = (3,3) # include a border arround of the field of width 3m
    meters_per_yard = 0.9144 # unit conversion from yards to meters
    half_pitch_length = field_dimen[0]/2. # length of half pitch
    half_pitch_width = field_dimen[1]/2. # width of half pitch
    signs = [-1,1]
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8*meters_per_yard
    box_width = 20*meters_per_yard
    box_length = 6*meters_per_yard
    area_width = 44*meters_per_yard
    area_length = 18*meters_per_yard
    penalty_spot = 12*meters_per_yard
    corner_radius = 1*meters_per_yard
    D_length = 8*meters_per_yard
    D_radius = 10*meters_per_yard
    D_pos = 12*meters_per_yard
    centre_circle_radius = 10*meters_per_yard
    # plot half way line # center circle
    ax.plot([0,0],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
    ax.scatter(0.0,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
    y = np.linspace(-1,1,50)*centre_circle_radius
    x = np.sqrt(centre_circle_radius**2-y**2)
    ax.plot(x,y,lc,linewidth=linewidth)
    ax.plot(-x,y,lc,linewidth=linewidth)
    for s in signs: # plots each line seperately
        # plot pitch boundary
        ax.plot([-half_pitch_length,half_pitch_length],[s*half_pitch_width,s*half_pitch_width],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
        # goal posts & line
        ax.plot( [s*half_pitch_length,s*half_pitch_length],[-goal_line_width/2.,goal_line_width/2.],pc+'s',markersize=6*markersize/20.,linewidth=linewidth)
        # 6 yard box
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[box_width/2.,box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[-box_width/2.,-box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*box_length,s*half_pitch_length-s*box_length],[-box_width/2.,box_width/2.],lc,linewidth=linewidth)
        # penalty area
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[area_width/2.,area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[-area_width/2.,-area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*area_length,s*half_pitch_length-s*area_length],[-area_width/2.,area_width/2.],lc,linewidth=linewidth)
        # penalty spot
        ax.scatter(s*half_pitch_length-s*penalty_spot,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
        # corner flags
        y = np.linspace(0,1,50)*corner_radius
        x = np.sqrt(corner_radius**2-y**2)
        ax.plot(s*half_pitch_length-s*x,-half_pitch_width+y,lc,linewidth=linewidth)
        ax.plot(s*half_pitch_length-s*x,half_pitch_width-y,lc,linewidth=linewidth)
        # draw the D
        y = np.linspace(-1,1,50)*D_length # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2-y**2)+D_pos
        ax.plot(s*half_pitch_length-s*x,y,lc,linewidth=linewidth)

    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0]/2. + border_dimen[0]
    ymax = field_dimen[1]/2. + border_dimen[1]
    ax.set_xlim([-xmax,xmax])
    ax.set_ylim([-ymax,ymax])
    ax.set_axisbelow(True)
    return fig,ax


# In[10]:


def plot_frame(hometeam, awayteam, homeplayers, awayplayers, homemapping, awaymapping, figax=None, team_colors=('r','b'), field_colour='green', field_dimen = (106.0,68.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False):
    """ plot_frame( hometeam, awayteam )

    Have re-written Laurie's plotting function to plot Signality data and to also deal with the numpy typing error that was occuring.
    Have also added team names, and the mapping from playerIndex number to the jersey number to allow for consistency of comparison of players between games.

    Parameters
    -----------
        hometeam: row (i.e. instant) of the home team tracking data frame
        awayteam: row of the away team tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)

    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """

    teamnames = [homeplayers.teamName[0], awayplayers.teamName[0]]

    homeIndexMapping = {homemapping[i]:i for i in homemapping}
    awayIndexMapping = {awaymapping[i]:i for i in awaymapping}

    homeAwayMappings = [homeIndexMapping, awayIndexMapping]

    if figax is None: # create new pitch
        fig,ax = plot_pitch(field_dimen = field_dimen, field_color=field_colour)
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple

    # plot home & away teams in order
    for team, color, name, mapping in zip( [hometeam, awayteam], team_colors, teamnames, homeAwayMappings) :

        x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
        y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions

        x = np.array(team[x_columns], dtype=np.float64)
        y = np.array(team[y_columns], dtype=np.float64)

        ax.plot( x, y, color+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha, label=name ) # plot player positions

        if include_player_velocities:
            vx_columns = np.array(['{}_vx'.format(c[:-2]) for c in x_columns]) # column header for player x positions
            vy_columns = np.array(['{}_vy'.format(c[:-2]) for c in y_columns]) # column header for player y positions

            vx = np.array(team[vx_columns], dtype=np.float64)
            vy = np.array(team[vy_columns], dtype=np.float64)

            ax.quiver( x, y, vx, vy, color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)

        if annotate:
            [ ax.text( team[x]+0.5, team[y]+0.5, mapping[int(x.split('_')[1])], fontsize=10, color=color  ) for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ]

    # plot ball
    ax.plot( hometeam['ball_x'], hometeam['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)

    return fig,ax


# In[11]:


def save_match_clip(hometeam, awayteam, fpath, fname='clip_test', figax=None, frames_per_second=25, team_colors=('r','b'), field_dimen = (106.0,68.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7):
    """ save_match_clip( hometeam, awayteam, fpath )

    Re-written Laurie's function to deal with bug caused by numpy typing of the (x,y) co-ords as it was fed into the quiver to plot the velocities
    Generates a movie from Signality tracking data, saving it in the 'fpath' directory with name 'fname'

    Parameters
    -----------
        hometeam: home team tracking data DataFrame. Movie will be created from all rows in the DataFrame
        awayteam: away team tracking data DataFrame. The indices *must* match those of the hometeam DataFrame
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.mp4'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Default is 0.7

    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    # check that indices match first
    assert np.all( hometeam.index==awayteam.index ), "Home and away team Dataframe indices must be the same"

    # in which case use home team index
    index = hometeam.index

    # Set figure and movie settings
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = fpath + '/' +  fname + '.mp4' # path and filename

    # create football pitch
    if figax is None:
        fig,ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig,ax = figax
    fig.set_tight_layout(True)

    # Generate movie
    print("Generating movie...",end='')
    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
            for team,color in zip( [hometeam.loc[i],awayteam.loc[i]], team_colors) :
                x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
                y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions

                x = np.array(team[x_columns], dtype=np.float64)
                y = np.array(team[y_columns], dtype=np.float64)

                objs, = ax.plot( x, y, color+'o', MarkerSize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
                figobjs.append(objs)
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions

                    vx = np.array(team[vx_columns], dtype=np.float64)
                    vy = np.array(team[vy_columns], dtype=np.float64)

                    objs = ax.quiver( x, y, vx, vy, color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
                    figobjs.append(objs)

            # plot ball
            objs, = ax.plot( team['ball_x'], team['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)
            figobjs.append(objs)

            # include match time at the top
            frame_minute =  int( team['Time [s]']/60. )
            frame_second =  ( team['Time [s]']/60. - frame_minute ) * 60.
            timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
            objs = ax.text(-2.5,field_dimen[1]/2.+1., timestring, fontsize=14 )
            figobjs.append(objs)
            writer.grab_frame()

            # Delete all axis objects (other than pitch lines) in preperation for next frame
            for figobj in figobjs:
                figobj.remove()

    print("\nMovie Completed.")
    plt.clf()
    plt.close(fig)


# # **4) Functions to Calculate Running Metrics**

# In[74]:


def summarise_match_running(df_players, df_tracks, playerJerseyMapping, pitchLength = 104.6, jogThreshold=2, runThreshold=4, sprintThreshold=7, plotDistances=True):
    """
    Builds on Laurie's code and uses his neat convolution trick to extract frame indices
    """

    playerJerseyMapping = {playerJerseyMapping[i]:i for i in playerJerseyMapping}

    df_summary = df_players.copy()

    # (1) Minutes played per player
    minutes = []
    for player in df_summary.playerIndex:
        # search for first and last frames that we have a position observation for each player (when a player is not on the pitch positions are NaN)
        column = f'Home_{player}_x' # use player x-position coordinate
        player_minutes = len(df_tracks.loc[pd.isna(df_tracks[column]) == False]) / (25*60)
        minutes.append( player_minutes )
    df_summary['Minutes Played'] = np.round(minutes, 2)

    # (2) Calculating total distance covered (essentialy integrating velocity over the constant timesteps)
    distance = []
    dt = 1/25
    for player in df_summary.playerIndex:
        column = f'Home_{player}_speed'
        player_distance = df_tracks[column].sum()*dt/1000
        distance.append(player_distance)
    df_summary['Distance [km]'] = np.round(distance, 2)

    # (3) Calculate distance covered while: walking, joggings, running, sprinting
    walking = []
    jogging = []
    running = []
    sprinting = []

    for player in df_summary.playerIndex:
        column = f'Home_{player}_speed'

        # walking (less than 2 m/s)
        player_distance = df_tracks.loc[df_tracks[column] < jogThreshold, column].sum()/25./1000
        walking.append(player_distance)

        # jogging (between 2 and 4 m/s)
        player_distance = df_tracks.loc[(df_tracks[column] >= jogThreshold) & (df_tracks[column] < runThreshold), column].sum()/25./1000
        jogging.append(player_distance)

        # running (between 4 and 7 m/s)
        player_distance = df_tracks.loc[(df_tracks[column] >= runThreshold) & (df_tracks[column] < sprintThreshold), column].sum()/25./1000
        running.append(player_distance)

        # sprinting (greater than 7 m/s)
        player_distance = df_tracks.loc[df_tracks[column] >= sprintThreshold, column].sum()/25./1000
        sprinting.append(player_distance)

    df_summary['Walking [km]'] = np.round(np.array(walking), 2)
    df_summary['Jogging [km]'] = np.round(np.array(jogging), 2)
    df_summary['Running [km]'] = np.round(np.array(running), 2)
    df_summary['Sprinting [km]'] = np.round(np.array(sprinting), 2)

    if plotDistances:
        ax = df_summary.sort_values('Distance [km]', ascending=False).plot.bar(x='jersey_number', y=['Walking [km]','Jogging [km]','Running [km]','Sprinting [km]'], colormap='coolwarm', figsize=(16,8))
        ax.set_xlabel('Jersey Number', fontsize=20)
        ax.set_ylabel('Distance covered [m]', fontsize=20)
        ax.legend(fontsize=20)

    ########################################################################################################
    ######       NOW WE CALCULATE THE SPECIFIC METRICS FOR THE FINAL PART OF THE ASSIGNEMENT          ######
    ########################################################################################################

    # sustained sprints: how many sustained sprints per match did each player complete? Defined as maintaining a speed > 7 m/s for at least 1 second
    nsprints = []
    nruns = []
    dic_runs = {}

    # minimum duration sprint should be sustained (in this case, 1 second = 25 consecutive frames)
    sprint_window = 25
    for player in df_players.playerIndex:
        column = f'Home_{player}_speed'
        column_x = f'Home_{player}_x'
        column_y = f'Home_{player}_y'

        ### LAURIE'S CONVOLUTION TRICK ###
        # trick here is to convolve speed with a window of size 'sprint_window', and find number of occassions that sprint was sustained for at least one window length
        # diff helps us to identify when the window starts
        player_sprints = np.diff( 1*( np.convolve( 1*(df_tracks[column]>=sprintThreshold), np.ones(sprint_window), mode='same' ) >= sprint_window ) )

        # to make sure runs and sprints are disjoint, let's handle them both with a lower limit, then take one from t'other at the end
        player_runs = np.diff( 1*( np.convolve( 1*(df_tracks[column]>=runThreshold), np.ones(sprint_window), mode='same' ) >= sprint_window ) )

        # counting the runs / sprints
        nsprints.append(np.sum(player_sprints == 1))
        nruns.append(np.sum(player_runs == 1))

        # getting the indices of the runs and sprints
        player_sprints_start = np.where( player_sprints == 1 )[0] - int(sprint_window/2) + 1 # adding sprint_window/2 because of the way that the convolution is centred
        player_sprints_end = np.where( player_sprints == -1 )[0] + int(sprint_window/2) + 1

        player_runs_start = np.where( player_runs == 1 )[0] - int(sprint_window/2) + 1 # adding sprint_window/2 because of the way that the convolution is centred
        player_runs_end = np.where( player_runs == -1 )[0] + int(sprint_window/2) + 1

        ## will now loop through the runs and sprints and figure out whether they were forward / backward / right / left, and whether they occured in the final third
        for n, (r_start, r_end) in enumerate(zip(player_runs_start, player_runs_end)):

            # getting the run delta y and delta x
            dy = df_tracks.loc[r_end, column_y] - df_tracks.loc[r_start, column_y]
            dx = df_tracks.loc[r_end, column_x] - df_tracks.loc[r_start, column_x]

            # getting the list of x coords so that we can see whether the run occured in the final third
            list_x = df_tracks.loc[r_start:r_end, column_x].values

            # initially started just looking at final third runs
            # but more interesting data when you look just in the opponents half
            final_third_threshold = 0#(pitchLength/3) - (pitchLength/2)
            final_third_flag = 1 if sum([1 if i < final_third_threshold else 0 for i in list_x]) > 0 else 0

            # classifying whether a run was left, right, back or forward
            if abs(dy) > abs(dx):
                if dy > 0:
                    run_direction = 'L'
                elif dy <= 0:
                    run_direction = 'R'
            elif abs(dx) >= abs(dy):
                if dx > 0:
                    run_direction = 'B'
                elif dx <= 0:
                    run_direction = 'F'

            dic_runs[f'{player}-{n+1}'] = [player, playerJerseyMapping[player], n+1, dy, dx, run_direction, final_third_flag, np.arange(r_start, r_end+1)]

    # transforming dic_runs dictionary -> dataframe
    dic_cols = ['playerIndex','jersey_number','runIndex','dy','dx','runDirection','finalThirdFlag','timeIndexArray']
    df_final_third = pd.DataFrame.from_dict(dic_runs, orient='index', columns=dic_cols)

    # just looking at runs in opponents half / final third
    df_final_third = df_final_third.loc[df_final_third['finalThirdFlag'] == 1]

    # summarising directional runs pivoting to produce directional run counts per player per match
    df_directional_runs = df_final_third.groupby(['jersey_number','runDirection'])\
                            .agg({'runIndex':'nunique'})\
                            .reset_index()\
                            .rename(columns={'runIndex':'numDirectionRuns'})\
                            .pivot(index='jersey_number', columns='runDirection', values='numDirectionRuns')\
                            .reset_index()

    df_summary['numSprints'] = nsprints
    df_summary['numRuns'] = nruns
    df_summary['numRuns'] = df_summary['numRuns'] - df_summary['numSprints']

    df_summary = df_summary.merge(df_directional_runs, on='jersey_number', how='inner')

    df_summary = df_summary.sort_values('Distance [km]', ascending=False)
    df_summary = df_summary[['jersey_number','name','Minutes Played','Distance [km]'\
                             ,'Walking [km]','Jogging [km]','Running [km]','Sprinting [km]'\
                             ,'numRuns','numSprints','B','F','L','R']].reset_index(drop=True)

    ########################################################################################################
    ######                                       SHEARING RUN  COMBINATIONS                           ######
    ########################################################################################################

    # producing a df of paired runs
    df_join = df_final_third.merge(df_final_third, on='finalThirdFlag', how='inner', suffixes=('_main','_other'))

    # only looking at paired runs in the final third
    df_join = df_join.loc[df_join['finalThirdFlag'] == 1]

    # getting rid of self runs, and removing dupes
    df_join = df_join.loc[(df_join['jersey_number_main'] < df_join['jersey_number_other'])]

    # looking at R/L or L/R runs
    df_join = df_join.loc[((df_join['runDirection_main'] == 'R') & (df_join['runDirection_other'] == 'L')) | ((df_join['runDirection_main'] == 'L') & (df_join['runDirection_other'] == 'R'))]

    # now getting the paired runs that overlapped with each other
    df_join['shearFrames'] = df_join.apply(lambda x: [i for i in x.timeIndexArray_main if i in x.timeIndexArray_other], axis=1)

    # getting the sync fraction of the shear run
    df_join['shearOverlapFraction'] = df_join.apply(lambda x: 2*len(x.shearFrames) / (len(x.timeIndexArray_main) + len(x.timeIndexArray_other)), axis=1)

    df_join['shearTime [s]'] = df_join.shearFrames.apply(lambda x: len(x)/25)

    df_join = df_join.loc[df_join['shearTime [s]'] > 0]

    df_shear = df_join.groupby(['jersey_number_main','jersey_number_other'])\
            .agg({'finalThirdFlag':np.sum, 'shearTime [s]':np.sum, 'shearOverlapFraction':lambda x: np.round(np.mean(x), 2)})\
            .reset_index()\
            .rename(columns={'finalThirdFlag':'numberShearCombos'})

    ########################################################################################################
    ######          SHEARING RUN COMBINATIONS THAT HAVE A THIRD FORWARD RUNNING PLAYER                ######
    ########################################################################################################

    # getting forward runs
    df_forward = df_final_third.loc[df_final_third['runDirection'] == 'F']

    # joining that forward run into the paired data frame
    df_forward = df_join.merge(df_forward, on='finalThirdFlag')

    # getting overlapping frames between forward runner and the shearing runs
    df_forward['forwardShearFrames'] = df_forward.apply(lambda x: [i for i in x.shearFrames if i in x.timeIndexArray], axis=1)

    # getting the time that the forward run overlaps with the shearing runs
    df_forward['shearTimeForward [s]'] = df_forward.forwardShearFrames.apply(lambda x: len(x)/25)

    # filtering out cases where there's no forward run overlapping with the shears
    df_forward = df_forward.loc[df_forward['shearTimeForward [s]'] > 0]

    # summarising
    df_forward_shear = df_forward.groupby(['jersey_number_main','jersey_number_other','jersey_number'])\
                    .agg({'finalThirdFlag':np.sum, 'shearTimeForward [s]':np.sum})\
                    .reset_index()\
                    .rename(columns={'finalThirdFlag':'numberShearCombos','jersey_number':'jersey_number_forward_runner'})

    return df_summary, df_shear, df_forward_shear


# ---
#
# &nbsp;
#
# &nbsp;
#
# &nbsp;
#
# # Will organise the code such that it goes through the assignment top to bottom.
#
# ## 1) **4 frame plots**

# ### 1.1) **Vs IF Elfsborg**

# In[14]:


# load tracking data for a given match
df_homePlayersElf, df_awayPlayersElf, df_homeTracksElf, df_awayTracksElf, pitchLengthElf, pitchWidthElf, homeJerseyMappingElf, awayJerseyMappingElf = \
parse_raw_to_df(signalityRepo, '20190722.Hammarby-IFElfsborg', interpolate=True)

# calculating velocities, accelerations, distances to goal and other metrics
df_homeTracksElf = calc_player_velocities(df_homeTracksElf, pitchLengthElf, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12)
df_awayTracksElf = calc_player_velocities(df_awayTracksElf, pitchLengthElf, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12)

# ### Elfsborg teamsheets

# In[15]:


df_homePlayersElf


# In[16]:


df_awayPlayersElf


# ### FIG 1

# In[17]:


frameIdx = 26828
fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)

fig.legend(loc='upper center', fontsize=17)

plt.savefig('Elfsborg_Goal_1.pdf', dpi=300, format='pdf', transparent=False, bbox_inches='tight')


# ### 1.2) **Vs Malmo FF**

# In[18]:


# load tracking data for a given match
df_homePlayersMal, df_awayPlayersMal, df_homeTracksMal, df_awayTracksMal, pitchLengthMal, pitchWidthMal, homeJerseyMappingMal, awayJerseyMappingMal = \
parse_raw_to_df(signalityRepo, '20191020.Hammarby-MalmöFF', interpolate=True)

df_homeTracksMal = calc_player_velocities(df_homeTracksMal, pitchLengthMal, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12)
df_awayTracksMal = calc_player_velocities(df_awayTracksMal, pitchLengthMal, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12)

# ### Malmo teamsheets

# In[19]:


df_homePlayersMal


# In[20]:


df_awayPlayersMal


# ### FIG 2

# In[21]:


frameIdx = 21032
fig, ax = plot_frame(df_homeTracksMal.loc[frameIdx], df_awayTracksMal.loc[frameIdx], df_homePlayersMal, df_awayPlayersMal, homeJerseyMappingMal, awayJerseyMappingMal,include_player_velocities=True, annotate=True, team_colors=('r','b'), field_colour='green', PlayerAlpha=0.8)

fig.legend(loc='upper center', fontsize=17)

plt.savefig('Malmo_Goal_1.pdf', dpi=300, format='pdf', transparent=False, bbox_inches='tight')


# ### 1.3) **Vs Orebro**

# In[22]:


# load tracking data for a given match
df_homePlayersOre, df_awayPlayersOre, df_homeTracksOre, df_awayTracksOre, pitchLengthOre, pitchWidthOre, homeJerseyMappingOre, awayJerseyMappingOre = \
parse_raw_to_df(signalityRepo, '20190930.Hammarby-Örebrö', interpolate=True)

df_homeTracksOre = calc_player_velocities(df_homeTracksOre, pitchLengthOre, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12)
df_awayTracksOre = calc_player_velocities(df_awayTracksOre, pitchLengthOre, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12)

# ### Orebro teamsheets

# In[23]:


df_homePlayersOre


# In[24]:


df_awayPlayersOre


# ### FIG 3

# In[25]:


frameIdx = 58300
fig, ax = plot_frame(df_homeTracksOre.loc[frameIdx], df_awayTracksOre.loc[frameIdx], df_homePlayersOre, df_awayPlayersOre, homeJerseyMappingOre, awayJerseyMappingOre,include_player_velocities=True, annotate=True, team_colors=('r','k'), field_colour='green', PlayerAlpha=0.8)

fig.legend(loc='upper center', fontsize=17)

plt.savefig('Orebro_Goal_1_scored.pdf', dpi=300, format='pdf', transparent=False, bbox_inches='tight')


# ### FIG 4

# In[26]:


frameIdx = 16220
fig, ax = plot_frame(df_homeTracksOre.loc[frameIdx], df_awayTracksOre.loc[frameIdx], df_homePlayersOre, df_awayPlayersOre, homeJerseyMappingOre, awayJerseyMappingOre,include_player_velocities=True, annotate=True, team_colors=('r','k'), field_colour='green', PlayerAlpha=0.8)

fig.legend(loc='upper center', fontsize=17)

plt.savefig('Orebro_Goal_1_conceded.pdf', dpi=300, format='pdf', transparent=False, bbox_inches='tight')


# ---
#
# &nbsp;
#
# &nbsp;
#
#
# ## **2) Insights & Limitations**
#
# > 15 second plots of distance from goal, speed, and acceleration for Hammarby over 15 second horizons.
#
# > Illustrate when the tracking data is accurate, and when the tracking data is less than accurate...

# ## 2.1) Mapping Errors

# ### **Reloading Velocities (no velocity smoothing or maxspeed cap)**

# In[202]:


df_homeTracksElf = calc_player_velocities(df_homeTracksElf, pitchLengthElf, smoothing=False, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 10000)
df_awayTracksElf = calc_player_velocities(df_awayTracksElf, pitchLengthElf, smoothing=False, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 10000)

# ### **Plotting Mapping Error**

# In[203]:


fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(50,42))

mappingErrorStart = 88300
mappingErrorFinish = mappingErrorStart + (15*25 - 1)

df_mappingErrorViz = df_homeTracksElf.loc[mappingErrorStart:mappingErrorFinish]

# code to help pick out specific jerseys
pattern = r'Home_(\d+)_'
homePlayerIndexMapping = {homeJerseyMappingElf[i]:i for i in homeJerseyMappingElf}

#specifying jerseys of interest
lstJerseysInterest = [5,7]

# getting x indices
x_ = (df_mappingErrorViz.index/(25*60))

D_cols = [i for i in df_mappingErrorViz if i[-1] == 'D']
v_cols = [i for i in df_mappingErrorViz if i[-5:] == 'speed']
a_cols = [i for i in df_mappingErrorViz if i[-12:] == 'acceleration']

for player in D_cols:

    jNumber = homePlayerIndexMapping[int(re.search(pattern, player).group(1))]
    pName = df_homePlayersElf.loc[df_homePlayersElf['jersey_number'] == jNumber].name.values[0]

    if jNumber in lstJerseysInterest:
        ax1.plot(x_, df_mappingErrorViz[player], lw=3)
    else:
        ax1.plot(x_, df_mappingErrorViz[player], alpha=0.4)

for player in v_cols:

    jNumber = homePlayerIndexMapping[int(re.search(pattern, player).group(1))]
    pName = df_homePlayersElf.loc[df_homePlayersElf['jersey_number'] == jNumber].name.values[0]

    if jNumber in lstJerseysInterest:
        ax2.plot(x_, df_mappingErrorViz[player], lw=3, label = f'{pName} (#{jNumber})')
    else:
        ax2.plot(x_, df_mappingErrorViz[player], alpha=0.4)

for player in a_cols:
    jNumber = homePlayerIndexMapping[int(re.search(pattern, player).group(1))]
    pName = df_homePlayersElf.loc[df_homePlayersElf['jersey_number'] == jNumber].name.values[0]

    if jNumber in lstJerseysInterest:
        ax3.plot(x_, df_mappingErrorViz[player], lw=3)
    else:
        ax3.plot(x_, df_mappingErrorViz[player], alpha=0.4)


ax1.set_ylabel(r'Distance from Goal (m)', fontsize=22)
ax2.set_ylabel(r'Speed (ms$^{-1}$)', fontsize=22)
ax3.set_ylabel(r'Acceleration (ms$^{-2}$)', fontsize=22)

# transforming ticks from min.min to min:secs
existingTicks = ax3.get_xticks()
newTicks = ["%02dm:%02ds" % (int(i), (i*60)%60) for i in existingTicks]
ax3.set_xticklabels(newTicks, fontsize=20)

# setting y-axis tick label size
ax1.set_yticklabels(ax1.get_yticks(), fontsize=20)
ax2.set_yticklabels(ax2.get_yticks(), fontsize=20)
ax3.set_yticklabels(ax3.get_yticks(), fontsize=20)

fig.legend(loc='center right', fontsize=24)


plt.savefig('MappingError.pdf', dpi=300, format='pdf', bbox_inches='tight')


# ### Photo Clips of Mapping Error

# In[204]:


frameIdx = 88401

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# In[205]:


frameIdx = 88402

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# In[206]:


frameIdx = 88403

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# ### Player mappings revert 100 frames later...

# In[207]:


frameIdx = 88501

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# In[208]:


frameIdx = 88503

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# In[209]:


frameIdx = 88505

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# ## **Combining 2.2) & 3) Nice view of breakaway goal: Goal 3 Vs Elfsborg**
#
# > Want another 15 second sequence that shows off the possibilities of tracking data
#
# > Will also combine this with the third part of the assignment to also calculate the distance to nearest teammate & opposition
#
# > And will start the focus on a handful of players to have a flowing narrative throughout the report

# ### **Reloading Velocities**

# In[210]:


df_homeTracksElf = calc_player_velocities(df_homeTracksElf, pitchLengthElf, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12)
df_awayTracksElf = calc_player_velocities(df_awayTracksElf, pitchLengthElf, smoothing=True, filter_='Savitzky-Golay', window=7, polyorder=1, maxspeed = 12)

# ### **Starting with frames for third goal Vs Elfsborg**

# In[211]:


frameIdx = 45020

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# In[212]:


frameIdx = 45080

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# In[213]:


frameIdx = 45105

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# In[214]:


frameIdx = 45130

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# In[215]:


frameIdx = 45175

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# In[216]:


frameIdx = 45213

fig, ax = plot_frame(df_homeTracksElf.loc[frameIdx], df_awayTracksElf.loc[frameIdx], df_homePlayersElf, df_awayPlayersElf, homeJerseyMappingElf, awayJerseyMappingElf,include_player_velocities=True, annotate=True, team_colors=('r','y'), field_colour='green', PlayerAlpha=0.8)


# ## **Khalili's second goal Vs IF Elfsborg: distance, velocity, acceleration, nearest teammate & opponent plots**

# In[217]:


# starting with the great goal viz dataframe and adding the away team
niceGoalStart = 45000
niceGoalFinish = niceGoalStart + (15*25 - 1)

df_homeGoal = df_homeTracksElf.loc[niceGoalStart:niceGoalFinish]
df_awayGoal = df_awayTracksElf.loc[niceGoalStart:niceGoalFinish]

def player_distance(x1, y1, x2, y2):
    """
    Function to calculate the distance between players
    """

    delta_x = x2-x1
    delta_y = y2-y1

    return np.sqrt(delta_x**2 + delta_y**2)

# home and away cols
homeCols = df_homeGoal.columns
awayCols = df_awayGoal.columns

# regex pattern to pick out positional cols
positionPattern = r'^(Home|Away)_(\d+)_[xy]'

#using default dicts to store the positional cols against the playerIndexes
dic_home = defaultdict(lambda: [])
dic_away = defaultdict(lambda: [])

# producing home & away dictionaries
## keys are playerIndices
## values are lists of the x and y column names for the positional cols
for h in homeCols:
    colMatch = re.match(positionPattern, h)
    if colMatch:
        dic_home[colMatch.group(2)].append(h)

for a in awayCols:
    colMatch = re.match(positionPattern, a)
    if colMatch:
        dic_away[colMatch.group(2)].append(a)

# combining home and away dataframes
df_homeAwayGoal = df_homeGoal\
                    .drop(columns=['ball_x','ball_y','ball_z','matchId','matchName','Period','Time [s]','ball_jerseyPossession','index','halfIndex','TimeStamp'])\
                    .merge(df_awayGoal, left_index=True, right_index=True, suffixes=('_home','_away'))

# adding four new columns per home player
## one for the nearest teammate distance
## one for nearest teammate playerIndex
## one for nearest opposition distance
## one for nearest opposition playerIndex
for player in dic_home:
    df_homeAwayGoal[f'Home_{player}_nearestTeammateDist'] = np.nan
    df_homeAwayGoal[f'Home_{player}_nearestTeammateIndex'] = np.nan
    df_homeAwayGoal[f'Home_{player}_nearestOppositionDist'] = np.nan
    df_homeAwayGoal[f'Home_{player}_nearestOppositionIndex'] = np.nan


##############################################################################################################################
##########                                      NEAREST NEIGHBOURS ALGORITHM                                        ##########
##############################################################################################################################

# looping through each frame
for idx, cols in df_homeAwayGoal.iterrows():

    # looping through each home player
    for player in dic_home:
        player_xCol, player_yCol = dic_home[player]
        player_x, player_y = cols[player_xCol], cols[player_yCol]

        # setting a max distance for teammate and opposition
        closestTeammateDistance = 1e6
        closestOppositionDistance = 1e6
        closestTeammate = 0
        closestOpposition = 0

        # looping through teammates
        for teammate in dic_home:

            # only interested in other teammates (otherwise you'll always be closest to yourself)
            if player != teammate:
                teammate_xCol, teammate_yCol = dic_home[teammate]
                teammate_x, teammate_y = cols[teammate_xCol], cols[teammate_yCol]
                teammateDist = player_distance(player_x, player_y, teammate_x, teammate_y)

                if teammateDist < closestTeammateDistance:
                    closestTeammateDistance = teammateDist
                    closestTeammate = teammate

        for opposition in dic_away:
            opposition_xCol, opposition_yCol = dic_away[opposition]
            opposition_x, opposition_y = cols[opposition_xCol], cols[opposition_yCol]
            oppositionDist = player_distance(player_x, player_y, opposition_x, opposition_y)

            if oppositionDist < closestOppositionDistance:
                closestOppositionDistance = oppositionDist
                closestOpposition = opposition

        df_homeAwayGoal.loc[idx, f'Home_{player}_nearestTeammateDist'] = closestTeammateDistance
        df_homeAwayGoal.loc[idx, f'Home_{player}_nearestTeammateIndex'] = closestTeammate
        df_homeAwayGoal.loc[idx, f'Home_{player}_nearestOppositionDist'] = closestOppositionDistance
        df_homeAwayGoal.loc[idx, f'Home_{player}_nearestOppositionIndex'] = closestOpposition

print ('Done.')


# ## Plotting

# In[220]:


fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(50,70))

# getting x indices
x_ = (df_homeGoal.index/(25*60))

# code to help pick out specific jerseys
pattern = r'Home_(\d+)_'
homePlayerIndexMapping = {homeJerseyMappingElf[i]:i for i in homeJerseyMappingElf}
awayPlayerIndexMapping = {awayJerseyMappingElf[i]:i for i in awayJerseyMappingElf}

#specifying jerseys of interest
lstJerseysInterest = [6,22,7,40]

# picking out columns for displacement to goal, speed, acceleration
D_cols = [i for i in df_homeGoal if i[-1] == 'D']
v_cols = [i for i in df_homeGoal if i[-5:] == 'speed']
a_cols = [i for i in df_homeGoal if i[-12:] == 'acceleration']

# filtering above columns for specific jerseys of interest
D_cols = [i for i in D_cols if homePlayerIndexMapping[int(re.search(pattern, i).group(1))] in lstJerseysInterest]
v_cols = [i for i in v_cols if homePlayerIndexMapping[int(re.search(pattern, i).group(1))] in lstJerseysInterest]
a_cols = [i for i in a_cols if homePlayerIndexMapping[int(re.search(pattern, i).group(1))] in lstJerseysInterest]

# plotting specific player distance to goal, speed, and acceleration
for player in D_cols:
    ax1.plot(x_, df_homeGoal[player], lw=3)

for player in v_cols:

    jNumber = homePlayerIndexMapping[int(re.search(pattern, player).group(1))]
    pName = df_homePlayersElf.loc[df_homePlayersElf['jersey_number'] == jNumber].name.values[0]
    ax2.plot(x_, df_homeGoal[player], lw=3, label = f'{pName} (#{jNumber})')

for player in a_cols:
    ax3.plot(x_, df_homeGoal[player], lw=3)



# getting the player labels for the below plots
closestTeammateLabels = [homePlayerIndexMapping[int(i)] for i in df_homeAwayGoal.Home_9_nearestTeammateIndex]
closestOppositionLabels = [awayPlayerIndexMapping[int(i)] for i in df_homeAwayGoal.Home_9_nearestOppositionIndex]

# Plotting Closest Teammates
ax4.scatter(x_, df_homeAwayGoal.Home_9_nearestTeammateDist, c=closestTeammateLabels, alpha=0.7)

# Labelling teammates on the chart
prevName = None
overUnder = -1
for i, j, k in zip(x_, df_homeAwayGoal.Home_9_nearestTeammateDist, closestTeammateLabels):

    pName = df_homePlayersElf.loc[df_homePlayersElf['jersey_number'] == k].name.values[0]

    if k != prevName:
        ax4.annotate(pName, (i, j+1.5*overUnder - 0.5), fontsize=18)
        overUnder *= -1

    prevName = k


# Plotting Opposition
ax5.scatter(x_, df_homeAwayGoal.Home_9_nearestOppositionDist, c=[np.log(i)*20000 for i in closestOppositionLabels], alpha=0.7)

# Labelling opposition players on the chart
prevName = None
overUnder = 1
for i, j, k in zip(x_, df_homeAwayGoal.Home_9_nearestOppositionDist, closestOppositionLabels):

    pName = df_awayPlayersElf.loc[df_awayPlayersElf['jersey_number'] == k].name.values[0]

    if k != prevName:
        ax5.annotate(pName, (i, j+1.5*overUnder - 3), fontsize=18)

    prevName = k


ax1.set_ylabel(r'Distance from Goal (m)', fontsize=22)
ax2.set_ylabel(r'Speed (ms$^{-1}$)', fontsize=22)
ax3.set_ylabel(r'Acceleration (ms$^{-2}$)', fontsize=22)
ax4.set_ylabel(r'Closest Teammate (m)', fontsize=22)
ax5.set_ylabel(r'Closest Opposition (m)', fontsize=22)

# transforming ticks from min.min to min:secs
existingTicks = ax5.get_xticks()
newTicks = ["%02dm:%02ds" % (int(i), (i*60)%60) for i in existingTicks]
ax5.set_xticklabels(newTicks, fontsize=20)

# setting y-axis tick label size
ax1.set_yticklabels(ax1.get_yticks(), fontsize=20)
ax2.set_yticklabels(ax2.get_yticks(), fontsize=20)
ax3.set_yticklabels(ax3.get_yticks(), fontsize=20)
ax4.set_yticklabels(ax4.get_yticks(), fontsize=20)
ax5.set_yticklabels(ax5.get_yticks(), fontsize=20)


# plotting the main actions

# pass
ax1.vlines(x=45080/(25*60), ymin=10, ymax=50, color='grey', alpha = 0.3)
ax2.vlines(x=45080/(25*60), ymin=0, ymax=6.5, label='Pass #6 -> #22', color='grey', alpha = 0.3)
ax3.vlines(x=45080/(25*60), ymin=0, ymax=6, color='grey', alpha = 0.3)
ax4.vlines(x=45080/(25*60), ymin=0, ymax=15, color='grey', alpha = 0.3)
ax5.vlines(x=45080/(25*60), ymin=0, ymax=17, color='grey', alpha = 0.3)

# assist
ax1.vlines(x=45105/(25*60), ymin=10, ymax=50, color='grey', alpha = 0.6)
ax2.vlines(x=45105/(25*60), ymin=0, ymax=6.5, label='Assist #22 -> #7', color='grey', alpha = 0.6)
ax3.vlines(x=45105/(25*60), ymin=0, ymax=6, color='grey', alpha = 0.6)
ax4.vlines(x=45105/(25*60), ymin=0, ymax=15, color='grey', alpha = 0.6)
ax5.vlines(x=45105/(25*60), ymin=0, ymax=17, color='grey', alpha = 0.6)

# goal
ax1.vlines(x=45175/(25*60), ymin=10, ymax=50, color='grey', alpha = 0.9)
ax2.vlines(x=45175/(25*60), ymin=0, ymax=6.5, label='Shot #7 -> Goal', color='grey', alpha = 0.9)
ax3.vlines(x=45175/(25*60), ymin=0, ymax=6, color='grey', alpha = 0.9)
ax4.vlines(x=45175/(25*60), ymin=0, ymax=15, color='grey', alpha = 0.9)
ax5.vlines(x=45175/(25*60), ymin=0, ymax=17, color='grey', alpha = 0.9)

fig.legend(loc='center right', fontsize=20)

plt.savefig('NiceGoalAdded.pdf', dpi=300, format='pdf', bbox_inches='tight')


# ## **4) Additional Run Metrics**
#
# **Building up our run statistics from simpler to more advanced statistics:**
# 1. Number of runs and sprints per player;
# 2. Number of runs and sprints broken down by run direction per player (forward, backward, left, right);
# 3. Number of shearing runs by pairs of players running left and right at the same time in the opponents half / final third;
# 4. Number of shearing runs pay pairs of players, where there's a forward run by a third player.

# ### **Vs Elfsborg**

# In[191]:


df_summaryElf, df_shearElf, df_forward_shearElf = summarise_match_running(df_homePlayersElf, df_homeTracksElf, homeJerseyMappingElf, pitchLengthElf)


# ### **Vs Malmo**

# In[192]:


df_summaryMal, df_shearMal, df_forward_shearMal = summarise_match_running(df_homePlayersMal, df_homeTracksMal, homeJerseyMappingMal, pitchLengthMal)


# ### **Vs Orebro**

# In[194]:


df_summaryOre, df_shearOre, df_forward_shearOre = summarise_match_running(df_homePlayersOre, df_homeTracksOre, homeJerseyMappingOre, pitchLengthOre)


# ### **Summarising per 90 mins over all games**

# In[221]:


df_summary = pd.concat([df_summaryElf, df_summaryMal, df_summaryOre], ignore_index=True)
df_summary['numMatches'] = 1
df_summary = df_summary.groupby(['jersey_number','name'])\
            .agg({'numMatches':np.sum,'Minutes Played':np.sum,'Distance [km]':np.sum,'Walking [km]':np.sum,'Jogging [km]':np.sum\
                 ,'Running [km]':np.sum, 'Sprinting [km]':np.sum, 'numRuns':np.sum,'numSprints':np.sum\
                 ,'B':np.sum,'F':np.sum,'L':np.sum,'R':np.sum})\
            .reset_index()

df_summary.iloc[:,4:14] = df_summary.iloc[:,4:14].div(df_summary.iloc[:,3], axis=0) * 90
df_summary['pcForward'] = 100*df_summary['F'] / (df_summary['F'] + df_summary['B'] + df_summary['R'] + df_summary['L'])
df_summary['pcSideToSide'] = 100*(df_summary['R'] + df_summary['L']) / (df_summary['F'] + df_summary['B'] + df_summary['R'] + df_summary['L'])

df_summary = df_summary.loc[df_summary['Minutes Played'] > 45].round(1).sort_values('pcForward', ascending=False).reset_index(drop=True)

df_summary


# ### Top five by percent of forward runs

# In[196]:


df_summary.round(1)[['name','F','B','L','R','pcForward']].head(5)



# ### Top five by percent of side-to-side runs

# In[228]:


df_summary.sort_values('pcSideToSide', ascending=False).round(1)[['name','F','B','L','R','pcSideToSide']].head(6)


# ### **Summarising the shearing runs over the three games**

# In[198]:


df_shear = pd.concat([df_shearElf, df_shearMal, df_shearOre], ignore_index=True)
df_shear = df_shear.groupby(['jersey_number_main','jersey_number_other'])\
            .agg({'numberShearCombos':np.sum,'shearTime [s]':np.sum,'shearOverlapFraction':np.mean})\
            .sort_values('numberShearCombos', ascending=False)\
            .reset_index()

df_shear[['jersey_number_main','jersey_number_other','numberShearCombos','shearTime [s]']].head(5)


# In[199]:
print ('Analysis Finished.')



# ---
#
# ## **End Technical Assignment Qs**
#
# ---
