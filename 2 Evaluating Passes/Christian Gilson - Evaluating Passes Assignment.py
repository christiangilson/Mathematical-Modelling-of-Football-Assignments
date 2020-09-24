#!/usr/bin/env python
# coding: utf-8

# ## Evaluating Passes Assignemnt
# In[1]:


# standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json
import os

# stats packages
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.calibration import calibration_curve


# plotting
from mplsoccer.pitch import Pitch

# to deal with the unicode characters of players names / team names in Wyscout
import codecs

pd.options.mode.chained_assignment = None

# # Helper Functions

# In[2]:


def show_event_breakdown(df_events, dic_tags):
    """
    Produces a full breakdown of the events, subevents, and the tags for the Wyscout dataset
    Use this to look at the various tags attributed to the event taxonomy
    """

    df_event_breakdown = df_events.groupby(['eventName','subEventName']).agg({'id':'nunique','tags':lambda x: list(x)}).reset_index().rename(columns={'id':'numSubEvents','tags':'tagList'})

    # creating a histogram of the tags per sub event
    df_event_breakdown['tagHist'] = df_event_breakdown.tagList.apply(lambda x: Counter([dic_tags[j] for i in x for j in i]))

    dic = {}

    for i, cols in df_event_breakdown.iterrows():
        eventName, subEventName, numEvents, tagList, tagHist = cols

        for key in tagHist:

            dic[f'{i}-{key}'] = [eventName, subEventName, numEvents, key, tagHist[key]]

    df_event_breakdown = pd.DataFrame.from_dict(dic, orient='index', columns=['eventName','subEventName','numSubEvents','tagKey','tagFrequency']).sort_values(['eventName','numSubEvents','tagFrequency'], ascending=[True, False, False]).reset_index(drop=True)
    return df_event_breakdown


# In[3]:


def home_and_away(df):
    """
    Picks out the home and away teamIds and their scores
    """
    teamsData = df['teamsData']

    for team in teamsData:
        teamData = teamsData[team]
        if teamData.get('side') == 'home':
            homeTeamId = team
            homeScore = teamData.get('score')
        elif teamData.get('side') == 'away':
            awayTeamId = team
            awayScore = teamData.get('score')

    df['homeTeamId'], df['homeScore'] = homeTeamId, homeScore
    df['awayTeamId'], df['awayScore'] = awayTeamId, awayScore

    return df


# In[4]:


def possession_indicator(df):
    """
    Function that identifies which team is in possession of the ball
    If the event is a found, interruption of offside, return a 0
    Winner of a duel is deemed in possession of the ball
    """

    # team identifiers
    teamId = df['teamId']
    homeTeamId = df['homeTeamId']
    awayTeamId = df['awayTeamId']
    teams = set([homeTeamId, awayTeamId])
    otherTeamId = list(teams - set([teamId]))[0]

    # eventName and subEventNames
    eventName = df['eventName']

    # success flag
    successFlag = df['successFlag']

    # assigning possession teamId
    if eventName in ['Pass','Free Kick','Others on the ball','Shot','Save attempt','Goalkeeper leaving line']:
        possessionTeamId = teamId
    elif eventName == 'Duel':
        possessionTeamId = teamId if successFlag == 1 else otherTeamId
    else:
        possessionTeamId = np.NaN

    return possessionTeamId


# In[5]:


def strong_foot_flag(df):
    """
    Compare foot of pass with footedness of player
    Provides flag = 1 if pass played with strong foot of the player
    """
    tags = df['tags']
    foot = df['foot']

    # tags
    if 401 in tags:
        passFoot = 'L'
    elif 402 in tags:
        passFoot = 'R'
    elif 403 in tags:
        passFoot = 'H'
    else:
        passFoot = 'N'

    # feature
    if (passFoot == 'L') and (foot in ['L','B']):
        strongFlag = 1
    elif (passFoot == 'R') and (foot in ['R','B']):
        strongFlag = 1
    else:
        strongFlag = 0

    return strongFlag


def weak_foot_flag(df):
    """
    Compare foot of pass with footedness of player
    Provides flag = 1 if pass played with weak foot of the player
    """
    tags = df['tags']
    foot = df['foot']

    # tags
    if 401 in tags:
        passFoot = 'L'
    elif 402 in tags:
        passFoot = 'R'
    elif 403 in tags:
        passFoot = 'H'
    else:
        passFoot = 'N'

    # feature
    if (passFoot == 'L') and (foot == 'R'):
        weakFlag = 1
    elif (passFoot == 'R') and (foot == 'L'):
        weakFlag = 1
    else:
        weakFlag = 0

    return weakFlag


# ---
#
# # Data Loader Functions
#
# * Players
# * Teams
# * Tags
# * Matches
# * Formations
# * Events

# In[6]:


def get_players(player_file):
    """
    Returns dataframe of players
    """

    with open(player_file) as f:
        players_data = json.load(f)

    player_cols = ['playerId','shortName','foot','height','weight','birthDate','birthCountry','role','roleCode']

    df_players = pd.DataFrame([[i.get('wyId'),codecs.unicode_escape_decode(i.get('shortName'))[0],i.get('foot'),i.get('height'),i.get('weight'),i.get('birthDate'),i.get('passportArea').get('name'),i.get('role').get('name'),i.get('role').get('code3')] for i in players_data], columns = player_cols)

    return df_players



def get_teams(team_file):
    """
    Returns dataframe of teams
    """

    with open(team_file) as f:
        teams_data = json.load(f)

    team_cols = ['teamId','teamName','officialTeamName','teamType','teamArea']

    df_teams = pd.DataFrame([[i.get('wyId'),codecs.unicode_escape_decode(i.get('name'))[0],codecs.unicode_escape_decode(i.get('officialName'))[0],i.get('type'),i.get('area').get('name')] for i in teams_data], columns=team_cols)

    return df_teams



dic_tags = {
     101: 'Goal',
     102: 'Own goal',
     301: 'Assist',
     302: 'Key pass',
     1901: 'Counter attack',
     401: 'Left foot',
     402: 'Right foot',
     403: 'Head/body',
     1101: 'Direct',
     1102: 'Indirect',
     2001: 'Dangerous ball lost',
     2101: 'Blocked',
     801: 'High',
     802: 'Low',
     1401: 'Interception',
     1501: 'Clearance',
     201: 'Opportunity',
     1301: 'Feint',
     1302: 'Missed ball',
     501: 'Free space right',
     502: 'Free space left',
     503: 'Take on left',
     504: 'Take on right',
     1601: 'Sliding tackle',
     601: 'Anticipated',
     602: 'Anticipation',
     1701: 'Red card',
     1702: 'Yellow card',
     1703: 'Second yellow card',
     1201: 'Position: Goal low center',
     1202: 'Position: Goal low right',
     1203: 'Position: Goal center',
     1204: 'Position: Goal center left',
     1205: 'Position: Goal low left',
     1206: 'Position: Goal center right',
     1207: 'Position: Goal high center',
     1208: 'Position: Goal high left',
     1209: 'Position: Goal high right',
     1210: 'Position: Out low right',
     1211: 'Position: Out center left',
     1212: 'Position: Out low left',
     1213: 'Position: Out center right',
     1214: 'Position: Out high center',
     1215: 'Position: Out high left',
     1216: 'Position: Out high right',
     1217: 'Position: Post low right',
     1218: 'Position: Post center left',
     1219: 'Position: Post low left',
     1220: 'Position: Post center right',
     1221: 'Position: Post high center',
     1222: 'Position: Post high left',
     1223: 'Position: Post high right',
     901: 'Through',
     1001: 'Fairplay',
     701: 'Lost',
     702: 'Neutral',
     703: 'Won',
     1801: 'Accurate',
     1802: 'Not accurate'
}



def get_matches(match_repo):
    """
    Return dataframe of matches
    """

    match_files = os.listdir(match_repo)

    lst_df_matches = []

    # note, this does not include groupName
    match_cols = ["status","roundId","gameweek","teamsData","seasonId","dateutc","winner","venue","wyId","label","date","referees","duration","competitionId","source"]

    for match_file in match_files:

        print (f'Processing {match_file}...')

        with open(f'matches/{match_file}') as f:
            data = json.load(f)
            df = pd.DataFrame(data)

            # adding some file source metadata
            df['source'] = match_file.replace('matches_','').replace('.json','')

            # dealing with the groupName column that's only in the international competitions
            df = df[match_cols]
            lst_df_matches.append(df)

    # concatenating match files
    df_matches = pd.concat(lst_df_matches, ignore_index=True)

    # applying home and away transformations using helper functions
    df_matches = df_matches.apply(home_and_away, axis=1)

    # and changing the wyId to matchId
    df_matches = df_matches.rename(columns={'wyId':'matchId'})

    # and filtering columns (may want to change this later)
    match_cols_final = ["source","competitionId","seasonId","roundId","gameweek","matchId","teamsData","dateutc","date","homeTeamId","homeScore","awayTeamId","awayScore","duration","winner","venue","label"]

    df_matches = df_matches[match_cols_final]

    return df_matches



def get_formations(df_matches):
    """
    Returns dataframe of formations within a match for all matches
    Adapted from https://github.com/CleKraus/soccer_analytics
    """

    lst_formations = list()

    for idx, match in df_matches.iterrows():

        matchId = match['matchId']

        # loop through the two teams
        for team in [0, 1]:
            team = match['teamsData'][list(match['teamsData'])[team]]
            teamId = team['teamId']

            # get all players that started on the bench
            player_bench = [player['playerId'] for player in team['formation']['bench']]
            df_bench = pd.DataFrame()
            df_bench['playerId'] = player_bench
            df_bench['lineup'] = 0

            # get all players that were in the lineup
            player_lineup = [
                player['playerId'] for player in team['formation']['lineup']
            ]
            df_lineup = pd.DataFrame()
            df_lineup['playerId'] = player_lineup
            df_lineup['lineup'] = 1

            # in case there were no substitutions in the match
            if team['formation']['substitutions'] == 'null':
                player_in = []
                player_out = []
                sub_minute = []
            # if there were substitutions
            else:
                player_in = [
                    sub['playerIn'] for sub in team['formation']['substitutions']
                ]
                player_out = [
                    sub['playerOut'] for sub in team['formation']['substitutions']
                ]
                sub_minute = [
                    sub['minute'] for sub in team['formation']['substitutions']
                ]

            # build a data frame who and when was substituted in
            df_player_in = pd.DataFrame()
            df_player_in['playerId'] = player_in
            df_player_in['substituteIn'] = sub_minute

            # build a data frame who and when was substituted out
            df_player_out = pd.DataFrame()
            df_player_out['playerId'] = player_out
            df_player_out['substituteOut'] = sub_minute

            # get the formation by concatenating lineup and bench players
            df_formation = pd.concat([df_lineup, df_bench], axis=0)
            df_formation['matchId'] = matchId
            df_formation['teamId'] = teamId

            # add information about substitutions
            df_formation = pd.merge(df_formation, df_player_in, how='left')
            df_formation = pd.merge(df_formation, df_player_out, how='left')

            lst_formations.append(df_formation)

    df_formations = pd.concat(lst_formations)

    # get the minute the player started and the minute the player ended the match
    df_formations['minuteStart'] = np.where(
        df_formations['substituteIn'].isnull(), 0, df_formations['substituteIn']
    )
    df_formations['minuteEnd'] = np.where(
        df_formations['substituteOut'].isnull(), 90, df_formations['substituteOut']
    )

    # make sure the match always lasts 90 minutes
    df_formations['minuteStart'] = np.minimum(df_formations['minuteStart'], 90)
    df_formations['minuteEnd'] = np.minimum(df_formations['minuteEnd'], 90)

    # set minuteEnd to 0 in case the player was not in the lineup and did not get substituted in
    df_formations['minuteEnd'] = np.where(
        (df_formations['lineup'] == 0) & (df_formations['substituteIn'].isnull()),
        0,
        df_formations['minuteEnd'],
    )

    # compute the minutes played
    df_formations['minutesPlayed'] = (
        df_formations['minuteEnd'] - df_formations['minuteStart']
    )

    # use a binary flag of substitution rather than a minute and NaNs
    df_formations['substituteIn'] = 1 * (df_formations['substituteIn'].notnull())
    df_formations['substituteOut'] = 1 * (df_formations['substituteOut'].notnull())

    return df_formations



def get_events(event_repo, leagueSelectionFlag = 0, leagueSelection = 'England'):
    """
    Returns dataframe of events
    """

    events_files = os.listdir(event_repo)

    lst_df_events = []

    if leagueSelectionFlag == 1:
        events_files = [i for i in events_files if i == f'events_{leagueSelection}.json']

    event_cols = ['source','matchId','matchPeriod','eventSec','teamId','id','eventId','eventName','subEventId','subEventName','playerId','positions','tags']

    for events_file in events_files:

        print (f'Processing {events_file}...')

        with open(f'events/{events_file}') as f:
            data = json.load(f)
            df = pd.DataFrame(data)
            df['source'] = events_file.replace('events_','').replace('.json','')
            lst_df_events.append(df)

    df_events = pd.concat(lst_df_events, ignore_index=True)

    # applying column re-ordering
    df_events = df_events[event_cols]

    return df_events


# ---
#
# # Event Feature Engineering Function
#
# * Rejigs tags
# * Applies `homeFlag`
# * Applies event `successFlag`
# * Applies `matchEventIndex` (an ordering of every event that occurs within a match from 1-n)
# * Applies `possessionTeamId` (the teamId that's in possession of the ball)
# * Applies `possessionSequenceIndex`
# * Applies `goalDelta` (the game state)
# * Applies `numReds` (the cumulative number of red cards a team has accrued throughout a match)
# * Applies `weakFlag` and `strongFlag` (for footedness of player and foot used for pass)
# * Unpacks `positions`
# * Applies `possessionStartSec`
# * Applies `playerPossessionTimeSec`
# * Re-orders and filters `df_events`

# In[7]:


def event_feature_engineering(df_events):
    """
    Takes in raw df_events dataframe and returns an augmented df_events dataframe with features for xPass model feature engineering
    """

    # Re-jigging tags -> list of integers
    print ('Rejigging tags...')
    df_events['tags'] = df_events.tags.apply(lambda x: [i.get('id') for i in x])


    # Applies homeFlag by 1) first merging on df_matches and then 2) applying helper function
    print ('Applying homeFlag...')
    ## 1)
    df_events = df_events.merge(df_matches, on=['matchId','source'], how = 'inner')
    ## 2)
    df_events['homeFlag'] = df_events.apply(lambda x: 1 if int(x.teamId) == int(x.homeTeamId) else 0, axis=1)


    # Applying success flag
    print ('Applying successFlag...')
    df_events['successFlag'] = df_events.tags.apply(lambda x: 1 if 1801 in x else 0)


    # 1) Ordering of events so that they're in precisely chronological order, and then 2) resorting (as the merge with df_matches will cause df_events to become unsorted)
    print ('Applying matchEventIndex...')
    ## 1)
    df_events['matchEventIndex'] = df_events.sort_values(['matchId','matchPeriod','eventSec'], ascending=[True, True, True])\
                                        .groupby('matchId')\
                                        .cumcount() + 1
    ## 2)
    df_events = df_events.sort_values(['matchId','matchEventIndex'], ascending=[True,True])


    # 1) Applying possession team indicator and then 2) forward filling the NaNs with the existing team (until possession is explicitly transferred)
    print ('Applying possessionTeamId...')
    ## 1)
    df_events['possessionTeamId'] = df_events.apply(possession_indicator, axis=1)
    ## 2) Filling the nans
    df_events['possessionTeamId'] = df_events.possessionTeamId.fillna(method='ffill')


    # Sequencing the possessions (each possession  will have it's own index per match)
    print ('Applying possessionSequenceIndex...')
    ## 1) initiate sequence at 0
    df_events['possessionSequenceIndex'] = 0
    ## 2) every time there's a change in sequence, you set a value of 1
    df_events['possessionSequenceIndex'][((df_events['possessionTeamId'] != df_events['possessionTeamId'].shift(1))) \
                                         | ((df_events['matchPeriod'] != df_events['matchPeriod'].shift(1)))] = 1
    ## 3) take a cumulative sum of the 1s per match
    df_events['possessionSequenceIndex'] = df_events.groupby('matchId')['possessionSequenceIndex'].cumsum()


    # Applying Game State
    ## Note this method is only 95% accurate; suspect that's sufficiently fine for this feature for this application
    print ('Applying gameState...')
    ## 1) getting goals scored flag
    df_events['goalScoredFlag'] = df_events.apply(lambda x: 1 if 101 in x.tags and x.eventName in ['Shot','Free Kick'] else 0, axis=1)
    ## 2) getting goal conceded flag
    df_events['goalsConcededFlag'] = df_events.apply(lambda x: 1 if 101 in x.tags and x.eventName == 'Save attempt' else 0, axis=1)
    ## 3) Cumulatively summing the goals scored
    df_events['goalsScored'] = df_events.sort_values(['matchId','matchPeriod','eventSec'], ascending=[True, True, True])\
                                        .groupby(['matchId','teamId'])\
                                        ['goalScoredFlag'].cumsum()
    ## 4) Cumulatively summing the goals conceded
    df_events['goalsConceded'] = df_events.sort_values(['matchId','matchPeriod','eventSec'], ascending=[True, True, True])\
                                        .groupby(['matchId','teamId'])\
                                        ['goalsConcededFlag'].cumsum()
    ## 5) Calculating the goal delta
    df_events['goalDelta'] = df_events['goalsScored'] - df_events['goalsConceded']


    # Applying red cards to calculate the difference in the number of players on each team
    print ('Applying numReds...')
    ## 1) Applying red card flag
    df_events['redCardFlag'] = df_events.tags.apply(lambda x: -1 if 1701 in x else 0)

    ## 2) Applying Excess Player flag to the other team
    df_reds = df_events.loc[df_events['redCardFlag'] == -1, ['matchId','teamId','matchEventIndex','id']]

    lst_redOtherTeamFlag = []

    for idx, cols in df_reds.iterrows():
        matchId, teamId, matchEventIndex, Id = cols
        try:
            redOtherTeamId = df_events.loc[(df_events['matchId'] == matchId) & (df_events['teamId'] != teamId) & (df_events['matchEventIndex'] > matchEventIndex)].sort_values('matchEventIndex', ascending=True)['id'].values[0]
            lst_redOtherTeamFlag.append(redOtherTeamId)
        except:
            continue

    df_events.loc[df_events['id'].isin(lst_redOtherTeamFlag), 'redCardFlag'] = 1

    ## 3) Cumulatively summing the number of red cards on a team throughout a game
    df_events['numReds'] = df_events.sort_values(['matchId','matchPeriod','eventSec'], ascending=[True, True, True])\
                                    .groupby(['matchId','teamId'])\
                                    ['redCardFlag'].cumsum()


    # Applying strong and weak foot flags
    print ('Applying weakFlag and strongFlag for footedness...')
    ## 1) adding player metadata
    df_events = df_events.merge(df_players, on='playerId', how='inner')
    ## 2) Cleaning up the foot preference flags of the players
    df_events['foot'] = df_events.foot.apply(lambda x: 'L' if x == 'left' else 'R' if x == 'right' else 'B' if x == 'both' else 'N')
    ## 3) Applying weak foot flag (mainly impacts crosses)
    df_events['weakFlag'] = df_events.apply(weak_foot_flag, axis=1)
    ## 4) Applying strong foot flag (this isn't seen as significant in the logistic regression, but keeping it in for completeness)
    df_events['strongFlag'] = df_events.apply(strong_foot_flag, axis=1)


    # Unpacking positions: Found that this multi-lambda method is by far and away the quickest rather than a multi-stage apply when dealing with 3M events
    print ('Unpacking positions...')
    # (this takes about a minute for the full Wyscout dataset which is pretty good)
    ## 1) counting the number of positions found in the position dic
    df_events['numPositions'] = df_events.positions.apply(lambda x: len(x))
    ## 2) Getting the starting x,y
    df_events['startPositions'] = df_events.positions.apply(lambda x: x[0])
    df_events['start_x'] = df_events.startPositions.apply(lambda x: x.get('x'))
    df_events['start_y'] = df_events.startPositions.apply(lambda x: x.get('y'))
    ## 3) Getting the ending x,y
    df_events['endPositions'] = df_events.apply(lambda x: x.positions[1] if x.numPositions == 2 else {}, axis=1)
    df_events['end_x'] = df_events.endPositions.apply(lambda x: x.get('x', None))
    df_events['end_y'] = df_events.endPositions.apply(lambda x: x.get('y', None))


    # Getting the time that the team has been in possession until the pass has been made (1) takes a while, but allows 2) to be vectorised)
    print ('Applying possessionStartSec...')
    ## 1) getting the time since the possession started
    df_events['possessionStartSec'] = df_events.loc[df_events.groupby(['matchId','possessionSequenceIndex'])['eventSec'].transform('idxmin'), 'eventSec'].values
    ## 2) calculating the time of the posession
    df_events['possessionTimeSec'] = df_events['eventSec'] - df_events['possessionStartSec']


    # Getting the time that the player has been in possession
    print ('Applying playerPossessionTimeSec...')
    ## 1) initialising at 0
    df_events['playerPossessionTimeSec'] = 0
    ## 2) checks that the previous event was part of the same possession sequence within the same match, and if it is, calculates possession time in seconds
    df_events['playerPossessionTimeSec'][((df_events['matchId'] == df_events['matchId'].shift(1)) &\
                                         (df_events['possessionSequenceIndex'] == df_events['possessionSequenceIndex'].shift(1)))]\
                                        = df_events['eventSec'] - df_events['eventSec'].shift(1)


    # Getting previous event
    print ('Grabbing previous event...')
    df_events['previousSubEventName'] = 'Match Start'
    df_events['previousSubEventName'][df_events['matchId'] == df_events['matchId'].shift(1)] = df_events['subEventName'].shift(1)


    # finally, tidying  up  columns
    df_events = df_events[['source','matchId','matchPeriod','eventSec','possessionTimeSec','playerPossessionTimeSec','matchEventIndex','teamId','homeTeamId','homeScore','awayTeamId','awayScore','homeFlag','id'\
                           ,'eventName','subEventName','previousSubEventName','possessionTeamId','possessionSequenceIndex','playerId','shortName','roleCode','strongFlag','weakFlag','goalDelta','numReds'\
                           ,'start_x','start_y','end_x','end_y','tags','successFlag']].sort_values(['matchId','matchEventIndex'], ascending=[True,True])

    print ('Outputting df_events.')
    return df_events


# In[8]:


def pass_feature_engineering(df_events, pitchLength = 105, pitchWidth = 68, outputToCsvFlag = 0):
    """
    Highly vectorised set of transformations

    Takes in the feature enriched df_events
    Filters on the different pass types
    Applies pass specific features
    Outputs df_passes
    """

    dic_passes = {
        'Simple pass':['Pass','Simple pass'],
        'High pass':['Pass','High pass'],
        'Head pass':['Pass','Head pass'],
        'Cross':['Pass','Cross'],
        'Launch':['Pass','Launch'],
        'Smart pass':['Pass','Smart pass'],
        'Hand pass':['Pass','Hand pass'],
        'Free kick cross':['Free Kick','Free kick cross'],
        'Corner':['Free Kick','Corner'],
        'Free Kick':['Free Kick','Free Kick'],
        'Throw in':['Free Kick','Throw in']
    }


    # Filtering df_events on relevant pass events
    ## 1) Applying filter
    print ('Applying pass filter...')
    df_passes = df_events.loc[df_events['subEventName'].isin(list(dic_passes.keys()))].copy()
    ## 2) DQ step: getting rid of two passes that don't have and end co-ord
    df_passes = df_passes.loc[pd.isna(df_passes['end_x']) == False].copy()


    # Series of geometric transformations
    print ('Applying geometric transformations...')
    ## 1) splitting the pitch into thirds and capturing the transition between thirds
    df_passes['startThird'] = df_passes.start_x.apply(lambda x: 1 if x < 34 else 2 if x < 67 else 3)
    df_passes['endThird'] = df_passes.end_x.apply(lambda x: 1 if x < 34 else 2 if x < 67 else 3)
    df_passes['thirdTransitionDelta'] = df_passes['endThird'] - df_passes['startThird']

    ## 2) transforming pitch dimensions from 100x100 grid to dimensions in meters
    df_passes['startPassM_x'] = df_passes.start_x*pitchLength/100
    df_passes['startPassM_y'] = df_passes.start_y*pitchWidth/100
    df_passes['endPassM_x'] = df_passes.end_x*pitchLength/100
    df_passes['endPassM_y'] = df_passes.end_y*pitchWidth/100

    ## 3) getting the squares of the x's
    df_passes['startPassM_xSquared'] = df_passes['startPassM_x']**2
    df_passes['endPassM_xSquared'] = df_passes['endPassM_x']**2

    ## 4) getting some central y stats and squared stats (same definitions as in David's code)
    df_passes['start_c'] = abs(df_passes['start_y'] - 50)
    df_passes['end_c'] = abs(df_passes['end_y'] - 50)
    df_passes['startM_c'] = df_passes['start_c']*pitchWidth/100
    df_passes['endM_c'] = df_passes['end_c']*pitchWidth/100
    df_passes['start_cSquared'] = df_passes['start_c']**2
    df_passes['end_cSquared'] = df_passes['end_c']**2
    df_passes['startM_cSquared'] = df_passes['startM_c']**2
    df_passes['endM_cSquared'] = df_passes['endM_c']**2

    ## 5) getting distance to ball
    df_passes['vec_x'] = df_passes['endPassM_x'] - df_passes['startPassM_x']
    df_passes['vec_y'] = df_passes['endPassM_y'] - df_passes['startPassM_y']
    df_passes['D'] = np.sqrt(df_passes['vec_x']**2 + df_passes['vec_y']**2)
    df_passes['Dsquared'] = df_passes.D**2
    df_passes['Dcubed'] = df_passes.D**3

    ## 6) DQ step: getting rid of events where the vec_x = vec_y = 0 (look like data errors)
    df_passes = df_passes.loc[~((df_passes['vec_x'] == 0) & (df_passes['vec_y'] == 0))].copy()

    ## 7) calculating passing angle in radians
    df_passes['a'] = np.arctan(df_passes['vec_x'] / abs(df_passes['vec_y']))
    #df_passes['aNew'] = np.arctan(df_passes['vec_x'] / (df_passes['endM_c'] - df_passes['startM_c']))

    ## 8) calculating shooting angle from initial position
    df_passes['aShooting'] = np.arctan(7.32 * df_passes['startPassM_x'] / (df_passes['startPassM_x']**2 + df_passes['startM_c']**2 - (7.32/2)**2))
    df_passes['aShooting'] = df_passes.aShooting.apply(lambda x: x+np.pi if x<0 else x)

    ## 9) calculating shooting angle from final position (i.e. )
    df_passes['aShootingFinal'] = np.arctan(7.32 * df_passes['endPassM_x'] / (df_passes['endPassM_x']**2 + df_passes['endM_c']**2 - (7.32/2)**2))
    df_passes['aShootingFinal'] = df_passes.aShootingFinal.apply(lambda x: x+np.pi if x<0 else x)

    ## 10) change in shooting angle caused by the pass
    df_passes['aShootingChange'] = df_passes['aShootingFinal'] - df_passes['aShooting']

    ## 11) distance to goal
    df_passes['DGoalStart'] = np.sqrt((pitchLength - df_passes['startPassM_x'])**2 + df_passes['startM_c']**2)
    df_passes['DGoalEnd'] = np.sqrt((pitchLength - df_passes['endPassM_x'])**2 + df_passes['endM_c']**2)
    df_passes['DGoalChange'] = df_passes['DGoalEnd'] - df_passes['DGoalStart']

    ## final) re-ordering cols
    df_passes = df_passes.sort_values(['matchId','matchEventIndex'], ascending=[True,True])



    # Within each possession sequence, applies the pass index (so the first pass in a possession is 1, and the second is 2, etc.)
    print ('Applying passIndexWithinSequence...')
    ## 1) produces index
    df_passes['passIndexWithinSequence'] = df_passes.sort_values(['matchId','possessionSequenceIndex','matchEventIndex'])\
                                                    .groupby(['matchId','possessionSequenceIndex'])\
                                                    .cumcount() + 1
    ## 2) LOOKAHEAD BIAS: WILL NOT INCLUDE THIS IN FINAL MODEL
    ## Calculating mean number of passes per possession per team
    df_meanNumPasses =  pd.DataFrame(df_passes.groupby(['teamId','possessionSequenceIndex'])\
                             .agg({'passIndexWithinSequence':np.mean})\
                             .groupby('teamId')\
                             .passIndexWithinSequence.mean())\
                            .reset_index()\
                            .rename(columns={'passIndexWithinSequence':'meanNumPassesPerSequence'})
    ## 3) Re-introducing this mean number of passes per possession via a join
    df_passes = df_passes.merge(df_meanNumPasses, how='inner', on='teamId')
    ## 4) getting the over under for the number of passes for that team
    ## COULD POTENTIALLY USE THIS IF HAD MULTIPLE YEARS OF HISTORY, AS IT SHOWS A CHARACTERISTIC OF A TEAM
    df_passes['numPassOverUnder'] = df_passes['passIndexWithinSequence'] - df_passes['meanNumPassesPerSequence']



    # Final set of flags (some are post-hoc so can't be used in the regression, but just adding for completeness)
    print ('Applying final set of flags...')
    ## 1) applying interception flag - this is of course highly correlated to an unsuccessful outcome, so won't be part of the regression
    df_passes['interceptionFlag'] = df_passes.tags.apply(lambda x: 1 if 1401 in x else 0)
    ## 2) applying dangerousBallLostFlag - this will also NOT be part of the regression
    df_passes['dangerousBallLostFlag'] = df_passes.tags.apply(lambda x: 1 if 2001 in x else 0)
    ## 3) counter attack flag
    df_passes['counterAttackFlag'] = df_passes.tags.apply(lambda x: 1 if 1901 in x else 0)
    ## 4) assist flag
    df_passes['assistFlag'] = df_passes.tags.apply(lambda x: 1 if 301 in x else 0)


    if outputToCsvFlag == 1:
        print ('Outputting df_passes to CSV...')
        df_passes.to_csv('df_passes.csv', index=None)

    print ('Outputting df_passes.')
    return df_passes


# ---
#
# # Model Application: Applying Four Models to Produce **xP** Variations to **Test** Data

# In[9]:


# applying basic, added, and advanced models to test data
def apply_xP_model_to_test(models):
    """
    Applying the four different models to produce four xP values
    """
    basic, added, adv_canonical, adv_probit = models

    print ('Applying models...')
    df_passes_test['xP_basic'] = basic.predict(df_passes_test)
    df_passes_test['xP_added'] = added.predict(df_passes_test)
    df_passes_test['xP_logit'] = adv_canonical.predict(df_passes_test)
    df_passes_test['xP'] = adv_probit.predict(df_passes_test)
    print (f'Done applying {len(models)} models.')

    return df_passes_test


# ---
#
# # Model Validation: Calibration Curves

# In[10]:


def plot_calibration_curve(df_passes_test, show_advanced=1, save_output=0):

    fig = plt.figure(figsize=(10, 10))

    # Plotting perfect calibration (line y=x)
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly Calibrated Model')

    alpha = 0.6
    numBins = 25

    # FOUR calibration curves - Tricky to plot all four at a time, so just do a Simple Vs Advanced
    if show_advanced == 0:
        ## 1) Simple Model
        fraction_of_positives, mean_predicted_value = calibration_curve(df_passes_test.successFlag, df_passes_test.xP_basic, n_bins=numBins)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Basic Model', alpha = alpha, color='red')

        ## 2) Added Model
        fraction_of_positives, mean_predicted_value = calibration_curve(df_passes_test.successFlag, df_passes_test.xP_added, n_bins=numBins)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Added Features', alpha = alpha, color='blue')

    elif show_advanced == 1:
        ## 3) Advanced Model: Canonical (Logit) Link function
        fraction_of_positives, mean_predicted_value = calibration_curve(df_passes_test.successFlag, df_passes_test.xP_logit, n_bins=numBins)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Advanced Features: Logit Link', alpha = alpha, color='black')

        ## 4) Advanced Model: Probit Link function
        fraction_of_positives, mean_predicted_value = calibration_curve(df_passes_test.successFlag, df_passes_test.xP, n_bins=numBins)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label='Advanced Features: Probit Link', alpha = alpha, color='orange')

    plt.ylabel('Fraction of Successful Passes', fontsize=18)
    plt.xlabel('Mean xP', fontsize=18)

    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])

    plt.legend(loc="lower right", fontsize=18)
    #plt.title('Calibration Plot', fontsize=24)

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.tight_layout()

    if save_output == 1:
        plt.savefig(f'calibration_{show_advanced}.pdf', dpi=300, format='pdf', bbox_inches='tight')

    return plt.show()


# ---
#
# # Model Validation: Metric Scores
#
# * Brier Score
# * Precision, Recall, F1
# * AUC
# * Accuracy

# In[11]:


def calculate_model_metrics(df_passes_test, xPtype='xP', log_reg_decision_threshold = 0.65):
    '''
    Applies Logistic Regression Decision Threshold (i.e. applying the model to attribute whether a pass would or would have not been successful)
    And calculates a bunch of related metrics
    '''

    df_passes_test['predictedSuccess'] = df_passes_test[xPtype].apply(lambda x: 1 if x > log_reg_decision_threshold else 0)

    brierScore = metrics.brier_score_loss(df_passes_test.successFlag, df_passes_test[xPtype])

    # precision = TRUE POSITIVE / (TRUE POSITIVE + FALSE POSITIVE)
    # ratio of correctly positive observations / all predicted positive observations
    precisionScore = metrics.precision_score(df_passes_test.successFlag, df_passes_test.predictedSuccess)

    # recall = TRUE POSITIVE / (TRUE POSITIVE + FALSE NEGATIVE)
    # ratio of correctly positive observations / all true positive observations (that were either correctly picked TP or missed FN)
    recallScore = metrics.recall_score(df_passes_test.successFlag, df_passes_test.predictedSuccess)

    # weighted average of precision and recall
    f1Score = metrics.f1_score(df_passes_test.successFlag, df_passes_test.predictedSuccess)

    AUCScore = metrics.roc_auc_score(df_passes_test.successFlag, df_passes_test.predictedSuccess)

    # overall accuracy score: ratio of all correct over count of all observations
    accuracyScore = metrics.accuracy_score(df_passes_test.successFlag, df_passes_test.predictedSuccess)

    return print (f'Brier Score: {brierScore}\n\nPrecision Score: {precisionScore}\n\nRecall Score: {recallScore}\n\nF1 Score: {f1Score}\n\nAUC Score: {AUCScore}\n\nAccuracyScore: {accuracyScore}')


# ---
#
# **CODE STARTS HERE**
#
# ---
#
#
# &nbsp;
#
# &nbsp;
#
# &nbsp;
#
# # 1) Loading Data
#
# ### Loading Players, Teams, Matches, Formations, Events

# In[12]:


df_players = get_players('players.json')
df_teams = get_teams('teams.json')
df_matches = get_matches('matches')
df_formations = get_formations(df_matches)
df_events = get_events('events', leagueSelectionFlag = 0, leagueSelection = 'England')


# ---
#
# # 2) Event Feature Engineering
#
# **Longest part of data preparation due to all of the nested feature extraction from the events data**:
#
# > Takes about 3 minutes if a single league is selected (`leagueSelectionFlag = 1` above).
#
# > Takes about 10 minutes if all leagues and international competitions are thrown into the mix.

# In[13]:


df_events = event_feature_engineering(df_events)


# ### For some bizarre reason, this code doesn't work if contained within the event feature engineering function, so adding it on here.
#
# (Doesn't effect modelling, only plotting.)

# In[14]:


# Getting the recipient player of an action (need to do this pre-pass filter as the next action may well not be a pass, and it'd be
## highly suboptimal to clip dangerous passes that resulted in shots and goals)
print ('Applying recipient of an event...')
possessionEventNames = ['Pass','Others on the ball','Shot']

df_events['passRecipientPlayerIdNext1'] = None
df_events['passRecipientPlayerIdNext2'] = None
df_events['passRecipientPlayerIdNext3'] = None
df_events['passRecipientPlayerIdNext4'] = None

df_events['passRecipientPlayerIdNext1'][((df_events['matchId'] == df_events['matchId'].shift(-1)) &\
                                    (df_events['matchEventIndex'] == (df_events['matchEventIndex'].shift(-1) - 1)) &\
                                    (df_events['possessionSequenceIndex'] == df_events['possessionSequenceIndex'].shift(-1)) &\
                                    (df_events['eventName'].shift(-1).isin(possessionEventNames)) &\
                                    (df_events['end_x'] == df_events['start_x'].shift(-1)) &\
                                    (df_events['end_y'] == df_events['start_y'].shift(-1)) &\
                                    (df_events['successFlag'] == 1))]\
                                        = df_events['playerId'].shift(-1)

df_events['passRecipientPlayerIdNext2'][((df_events['matchId'] == df_events['matchId'].shift(-2)) &\
                                    (df_events['matchEventIndex'] == (df_events['matchEventIndex'].shift(-2) - 2)) &\
                                    (df_events['possessionSequenceIndex'] == df_events['possessionSequenceIndex'].shift(-2)) &\
                                    (df_events['eventName'].shift(-2).isin(possessionEventNames)) &\
                                    (df_events['end_x'] == df_events['start_x'].shift(-2)) &\
                                    (df_events['end_y'] == df_events['start_y'].shift(-2)) &\
                                    (df_events['successFlag'] == 1))]\
                                        = df_events['playerId'].shift(-2)

df_events['passRecipientPlayerIdNext3'][((df_events['matchId'] == df_events['matchId'].shift(-3)) &\
                                    (df_events['matchEventIndex'] == (df_events['matchEventIndex'].shift(-3) - 3)) &\
                                    (df_events['possessionSequenceIndex'] == df_events['possessionSequenceIndex'].shift(-3)) &\
                                    (df_events['eventName'].shift(-3).isin(possessionEventNames)) &\
                                    (df_events['end_x'] == df_events['start_x'].shift(-3)) &\
                                    (df_events['end_y'] == df_events['start_y'].shift(-3)) &\
                                    (df_events['successFlag'] == 1))]\
                                        = df_events['playerId'].shift(-3)

df_events['passRecipientPlayerIdNext4'][((df_events['matchId'] == df_events['matchId'].shift(-4)) &\
                                    (df_events['matchEventIndex'] == (df_events['matchEventIndex'].shift(-4) - 4)) &\
                                    (df_events['possessionSequenceIndex'] == df_events['possessionSequenceIndex'].shift(-4)) &\
                                    (df_events['eventName'].shift(-4).isin(possessionEventNames)) &\
                                    (df_events['end_x'] == df_events['start_x'].shift(-4)) &\
                                    (df_events['end_y'] == df_events['start_y'].shift(-4)) &\
                                    (df_events['successFlag'] == 1))]\
                                        = df_events['playerId'].shift(-4)


df_events['passRecipientPlayerId'] = df_events.apply(lambda x: int(x.passRecipientPlayerIdNext1) if x.passRecipientPlayerIdNext1 != None else\
                                                    int(x.passRecipientPlayerIdNext2) if x.passRecipientPlayerIdNext2 != None else\
                                                    int(x.passRecipientPlayerIdNext3) if x.passRecipientPlayerIdNext3 != None else\
                                                    int(x.passRecipientPlayerIdNext4) if x.passRecipientPlayerIdNext4 != None else None, axis=1)

# ---
#
# # 3) Pass Specific Feature Engineering
#
# **Highly vectorised, so only takes a minute with all leagues loaded in**
#

# In[15]:


df_passes = pass_feature_engineering(df_events, outputToCsvFlag=0)


# ---
#
# # 4) Model Fitting
#
# ### Splitting `df_passes` into training and test dataset, stratifying the dependent variable

# In[16]:


# splitting into a dataframe for training and dataframe for testing
## stratifying the successFlag
df_passes_train, df_passes_test = train_test_split(df_passes, test_size=0.3, stratify=df_passes.successFlag, random_state=1, shuffle=True)

print (f'Stratified Pass Success Rates:\n\nOverall: {100*np.round(df_passes.successFlag.mean(),3)}%\nTrain: {100*np.round(df_passes_train.successFlag.mean(), 3)}%\nTest: {100*np.round(df_passes_test.successFlag.mean(), 3)}%\n')


# ### Fitting basic model to **training** data:
# * Starting X
# * Starting Y

# In[17]:


pass_model_basic = smf.glm(formula="successFlag ~ startPassM_x + startM_c", data=df_passes_train\
                 ,family=sm.families.Binomial()).fit()

#pass_model_basic.summary2()


# ### Fitting addititional features:
# * Starting X
# * Starting Y
# * X\*Y (Interaction Term)
# * End X
# * End Y
# * Shooting Angle (Initial)
# * Sub Event Type

# In[18]:


pass_model_added = smf.glm(formula="successFlag ~ C(subEventName) + startPassM_x*startM_c + endPassM_x + endM_c + aShooting +\
                    startPassM_xSquared + startM_cSquared", data=df_passes_train\
                 ,family=sm.families.Binomial()).fit()

#pass_model_added.summary2()


# ### Fitting model to **training** data with advanced features, using two different link functions:
#
# #### Features:
#
# * Sub Event Name
# * Starting X
# * Starting Y
# * Starting X\*Y (Interaction Term)
# * End X
# * End Y
# * End X\*Y (Interaction Term)
# * Start Y^2
# * End Y^2
# * Start Y^2 \* End Y^2 (Interaction Term)
# * Start X^2
# * End X^2
# * Start X^2 \* End X^2 (Interaction Term)
# * Distance to Goal (Initial)
# * Passing Distance
# * Passing Distance^2
# * Passing Distance^3
# * Passing Angle
# * Shooting Angle (Initial)
# * Shooting Angle (Change Before and After Pass)
# * Transition Through Thirds (1->2, 2->3, etc.)
# * Home / Away Flag
# * Counter Attack Flag
# * Number of Red Cards
# * Game State (Delta Between Teams for Number of Goals Scored)
# * Time of Current Possession Sequence
# * Time of Passing Player Possession
# * Passing Index Within Possession Sequence
#
#
#
# #### Link Functions:
# * Logit (Canonical link function for Binomial family of distributions)
# * Probit
#

# In[19]:


pass_model_advanced_canonical = smf.glm(formula="successFlag ~ C(eventName) + C(subEventName) +\
                      startPassM_x*startM_c + endPassM_x*endM_c + start_cSquared*end_cSquared +\
                      startPassM_xSquared*endPassM_xSquared +\
                      D + DGoalStart + Dsquared + Dcubed +\
                      a + aShooting + aShootingChange +\
                      thirdTransitionDelta +\
                      C(homeFlag) + C(counterAttackFlag) +\
                      numReds + goalDelta +\
                      possessionTimeSec + playerPossessionTimeSec + passIndexWithinSequence", data=df_passes_train\
                 ,family=sm.families.Binomial(link=sm.families.links.logit)).fit()

#pass_model_advanced_canonical.summary2()

# In[20]:


pass_model_advanced_probit = smf.glm(formula="successFlag ~ C(eventName) + C(subEventName) +\
                      startPassM_x*startM_c + endPassM_x*endM_c + start_cSquared*end_cSquared +\
                      startPassM_xSquared*endPassM_xSquared +\
                      D + DGoalStart + Dsquared + Dcubed +\
                      a + aShooting + aShootingChange +\
                      thirdTransitionDelta +\
                      C(homeFlag) + C(counterAttackFlag) +\
                      numReds + goalDelta +\
                      possessionTimeSec + playerPossessionTimeSec + passIndexWithinSequence", data=df_passes_train\
                 ,family=sm.families.Binomial(link=sm.families.links.probit)).fit()

#pass_model_advanced_probit.summary2()


# ---
#
# # 5) Applying models to **test** data
#
# 1. Basic model: just starting position features;
# 2. Added model: including features outlined in the problem statement;
# 3. Advanced model: Logit link - added extra **x** features
# 4. Advabced model: Probit link (same features as above)

# In[21]:


df_passes_test = apply_xP_model_to_test([pass_model_basic, pass_model_added, pass_model_advanced_canonical, pass_model_advanced_probit])

# ---
#
#
# # 6) Model Validation: Calibration Curves of Models Fit to **Test** Data

# ### Calibration Curve: Basic Vs Added Models

# In[22]:


plot_calibration_curve(df_passes_test, show_advanced=0, save_output=1)

# ### Calibration Curve - Advanced Models: Logit Vs Probit

# In[23]:


plot_calibration_curve(df_passes_test, show_advanced=1, save_output=1)

# ---
#
# &nbsp;
#
# &nbsp;
#
# &nbsp;
#
# # 7) Applying Logistic Regression Classifier and Calculating Model Fit Metrics

# In[24]:


calculate_model_metrics(df_passes_test, 'xP')

# ---
#
# # 8) Applying advanced model to `df_passes` (**test + training** data) and **summarising best forwards at "risky" passes in England**

# In[25]:


leagues = ['England'] #['England','Spain','Italy','Germany','France']
positions = ['FWD']
xPthreshold = 0.5
minMatches = 10
minPasses = 50


# In[26]:


# applying advanced model to predict xP
df_passes['xP'] = pass_model_advanced_probit.predict(df_passes)
# calculating "overxP"
df_passes['overxP'] = df_passes['successFlag'] - df_passes['xP']

## get average distance and average y distance metrics, too

df_overall_pcSuccess = df_passes.loc[df_passes['source'].isin(leagues)]\
                                .groupby(['playerId','shortName'])\
                                .agg({'successFlag':np.sum,'id':'nunique'})\
                                .rename(columns={'successFlag':'overallSuccessful','id':'overallAttempted'})

df_overall_pcSuccess['overallPcCompleted'] = np.round(100*df_overall_pcSuccess['overallSuccessful'] / df_overall_pcSuccess['overallAttempted'], 1)

# producing summary, adding in formations data to calculate the excess xP per 90 minutes
df_summary_passer = df_passes.loc[(df_passes['roleCode'].isin(positions))\
                           & (df_passes['source'].isin(leagues))\
                           & (df_passes['xP'] < xPthreshold)]\
        .merge(df_formations, on=['matchId','teamId','playerId'], how='inner')\
        .groupby(['matchId','playerId','roleCode','minutesPlayed','shortName'])\
        .agg({'overxP':np.sum,'id':'nunique','successFlag':np.sum,'vec_x':np.sum})\
        .reset_index()\
        .rename(columns={'id':'totAttemptedPerMatch','successFlag':'totSuccessfulPerMatch'})\
        .groupby(['playerId','shortName','roleCode'])\
        .agg({'overxP':np.sum,'totAttemptedPerMatch':np.sum,'totSuccessfulPerMatch':np.sum,'minutesPlayed':np.sum,'matchId':'nunique','vec_x':np.sum})\
        .rename(columns={'totAttemptedPerMatch':'totAttempted','totSuccessfulPerMatch':'totSuccessful','matchId':'totMatches'})\
        .sort_values('overxP', ascending=False)\
        .reset_index()

# getting the overall fraction of completed passes
df_summary_passer['pcCompleted'] = np.round(100*df_summary_passer['totSuccessful'] / df_summary_passer['totAttempted'], 1)
# getting a normalised metric per 90 minutes of play
df_summary_passer['overxPper90mins'] = np.round(90*(df_summary_passer['overxP'] / df_summary_passer['minutesPlayed']), 1)
# getting a normalised metric per 100 attempts
df_summary_passer['overxPper100attempts'] = np.round(100*(df_summary_passer['overxP'] / df_summary_passer['totAttempted']), 1)
# explicitly making mins played an integer
df_summary_passer['minutesPlayed'] = df_summary_passer.minutesPlayed.apply(lambda x: int(x))
# rounding overxP score
df_summary_passer['overxP'] = np.round(df_summary_passer['overxP'], 1)

# joining difficult pass table to overall pc completed table
df_summary_passer = df_summary_passer.merge(df_overall_pcSuccess, on=['playerId','shortName'], how='inner')
df_summary_passer['pcTrickyBall'] = np.round(100*df_summary_passer['totAttempted'] / df_summary_passer['overallAttempted'], 1)

# filtering using minimum criteria
df_summary_passer = df_summary_passer.loc[(df_summary_passer['totMatches'] >= minMatches) & (df_summary_passer['totAttempted'] >= minPasses)]

df_summary_passer = df_summary_passer[['playerId','shortName','roleCode','overxP','overxPper90mins','overxPper100attempts','totSuccessful','totAttempted','pcCompleted','minutesPlayed','totMatches','pcTrickyBall','overallPcCompleted','vec_x']]

df_summary_passer.sort_values('overxPper100attempts', ascending=False).head(40)




# ## Hazard distribution visualisation

# In[28]:


df_hazard = df_passes.loc[(df_passes['playerId'] == 25707) & (df_passes['source'] == 'England')\
                          & (df_passes['xP'] < 0.5)]

df_hazard = df_hazard.merge(df_players.rename(columns={'playerId':'passRecipientPlayerId'}), on='passRecipientPlayerId', how='inner', suffixes=(['_hazard','_receiver']))

hazard_main_receivers = ['Álvaro Morata','Willian','Pedro','Fàbregas','V. Moses','O. Giroud']

df_hazard_main_receivers = df_hazard.loc[df_hazard['shortName_receiver'].isin(hazard_main_receivers)][['source','matchId','eventSec','subEventName','previousSubEventName','shortName_hazard','shortName_receiver','xP','overxP','startPassM_x','startPassM_y','endPassM_x','endPassM_y']]


# ## Plotting Hazard Passes by Receiver

# In[29]:


pitch = Pitch(layout=(3, 2), tight_layout=False, constrained_layout=True, view='half', orientation='vertical', figsize=(12,12))

fig, axs = pitch.draw()

for ax, receiver in zip(axs.flat, hazard_main_receivers):

    df_receiver = df_hazard_main_receivers.loc[df_hazard_main_receivers['shortName_receiver'] == receiver]

    # smart passes
    pitch.lines(df_receiver.loc[df_receiver['subEventName'] == 'Smart pass'].startPassM_x*120/105, df_receiver.loc[df_receiver['subEventName'] == 'Smart pass'].startPassM_y*80/68,\
                df_receiver.loc[df_receiver['subEventName'] == 'Smart pass'].endPassM_x*120/105, df_receiver.loc[df_receiver['subEventName'] == 'Smart pass'].endPassM_y*80/68,
                lw=10, transparent=True, comet=True,
                label=f'Smart passes to {receiver}', color='red', ax=ax, alpha_start=0.01, alpha_end=1)

    # crosses
    pitch.lines(df_receiver.loc[df_receiver['subEventName'] == 'Cross'].startPassM_x*120/105, df_receiver.loc[df_receiver['subEventName'] == 'Cross'].startPassM_y*80/68,\
                df_receiver.loc[df_receiver['subEventName'] == 'Cross'].endPassM_x*120/105, df_receiver.loc[df_receiver['subEventName'] == 'Cross'].endPassM_y*80/68,
                lw=10, transparent=True, comet=True,
                label=f'Crosses to {receiver}', ax=ax, alpha_start=0.01, alpha_end=1)

    leg = ax.legend(borderpad=1, markerscale=1.5, labelspacing=1.5, loc='lower left', fontsize=12)

fig.savefig('Hazard.pdf', format='pdf',dpi=300,pad_inches=0,bbox_inches='tight', transparent=True)
