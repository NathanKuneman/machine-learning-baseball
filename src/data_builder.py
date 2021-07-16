from pybaseball.lahman import *
from pybaseball.lahman import pitching as pitch
from pybaseball import playerid_reverse_lookup
import pandas as pd
import numpy as np
import os
import warnings


class Constructor:

    def __init__(self, file_directory):
        self.file_directory = file_directory
        self.career_totals = self.career_numbers()
        self.pitcher_df = self.pitching_career_totals()

        columns = list(self.career_totals.columns)
        columns.extend(['fantasy_ppg', 'starting_pitcher_hand', 'ppg_vs_hand', 'fantasy_points', 'season', 'singles_', 'doubles_', 'triples_', 'home_runs_', 'rbis_', 'runs_', 'walks_', 'stolen_bases_', 'hbp_', 'ab_'])

        
        self.batter_df = pd.DataFrame(columns=columns)
        
        self.ballpark_factors_hr = {'ANA': 1.11,
                                    'ARI': .82,
                                    'ATL': 1,
                                    'BAL': 1.19,
                                    'BOS': 1.02,
                                    'CHA': 1.18,
                                    'CHN': 1.05,
                                    'CIN': 1.25,
                                    'CLE': 1.04,
                                    'COL': 1.10,
                                    'DET': .85,
                                    'FLO': .85,
                                    'MIA': .85,
                                    'HOU': 1.01,
                                    'KCA': .79,
                                    'LAN': 1.12,
                                    'MIL': 1.06,
                                    'MIN': .94,
                                    'NYA': 1.17,
                                    'NYN': .95,
                                    'OAK': .87,
                                    'PHI': 1.07,
                                    'PIT': .84,
                                    'SDN': .97,
                                    'SEA': 1.06,
                                    'SFN': .85,
                                    'SLN': .87,
                                    'TBA': .92,
                                    'TEX': .95,
                                    'TOR': 1.05,
                                    'WAS': 1.01}

        self.ballpark_factors_ba = {'ANA': .98,
                                    'ARI': 1.02,
                                    'ATL': 1.01,
                                    'BAL': 1.01,
                                    'BOS': 1.05,
                                    'CHA': .99,
                                    'CHN': 1.03,
                                    'CIN': 1.02,
                                    'CLE': 1.03,
                                    'COL': 1.15,
                                    'DET': 1.0,
                                    'FLO': .98,
                                    'MIA': .98,
                                    'HOU': .97,
                                    'KCA': 1.03,
                                    'LAN': 1,
                                    'MIL': .99,
                                    'MIN': 1.01,
                                    'NYA': .98,
                                    'NYN': .94,
                                    'OAK': .96,
                                    'PHI': .98,
                                    'PIT': 1,
                                    'SDN': .98,
                                    'SEA': .94,
                                    'SFN': 1.01,
                                    'SLN': .99,
                                    'TBA': .95,
                                    'TEX': .99,
                                    'TOR': 1.02,
                                    'WAS': .99}
        
    

    def career_numbers(self):

        download_lahman()
        batting_numbers = batting()
        recent_years = batting_numbers[batting_numbers['yearID'] > 1980]

        # Limit to players who have played since 1980
        career_totals = recent_years.groupby('playerID').sum()
        career_totals.reset_index()

        # Limit to only players with 100+ AB
        career_totals = career_totals[career_totals['AB'] > 100]

        # Set career stat columns
        # Batting Average
        career_totals['BA'] = career_totals['H'] / career_totals['AB']
        # 2B/AB
        career_totals['2B/AB'] = career_totals['2B'] / career_totals['AB']
        # 3B/AB
        career_totals['3B/AB'] = career_totals['3B'] / career_totals['AB']
        # HR/AB
        career_totals['HR/AB'] = career_totals['HR'] / career_totals['AB']
        # RBI/G
        career_totals['RBI/G'] = career_totals['RBI'] / career_totals['G']
        # R/G
        career_totals['R/G'] = career_totals['R'] / career_totals['G']
        # BB/G
        career_totals['BB/G'] = career_totals['BB'] / career_totals['G']
        # SB/G
        career_totals['SB/G'] = career_totals['SB'] / career_totals['G']
        # HBP/G
        career_totals['HBP/G'] = career_totals['HBP'] / career_totals['G']


        # Drop columns we used to create our new columns 
        career_totals = career_totals.iloc[:,19:]
        career_totals = career_totals.reset_index()

        return career_totals
       
        
    
    def create_training_dataframe(self):

        # Create a blank Dataframe that has all available columns 
        columns = list(self.career_totals.columns)
        columns.append('fantasy_points')
        files = os.listdir('../data/processed_events')
        num_files = len(files)
        count = 1
        for file_path in files[148:]:
            complete_path = '../data/processed_events/' + file_path
            print(f'Current file is {file_path}')
            self.add_game_data(complete_path)
            print(f'Completed {count} files of {num_files}')
            count += 1
            self.save_dataframe('../data/completed_df_part2.csv')
            
        self.batter_df = pd.get_dummies(self.batter_df, columns=['position'], drop_first=True)


    def add_game_data(self, complete_path):
        event_file = pd.read_csv(complete_path, header=None)
        columns = {0:'game_id', 1:'batter', 2:'batter_hand', 3:'pitcher', 4:'pitcher_hand',
        5:'run_first', 6:'run_second', 7:'run_third', 8:'batter_pos', 9:'result', 10: 'ab',
        11:'rbi', 12:'first_dest', 13:'second_dest', 14:'third_dest', 15:'sb_first',
        16:'sb_second', 17:'sb_third'}

        event_file = event_file.rename(columns=columns)
        games = event_file['game_id'].unique()
        for game in games:
            current_game = event_file[event_file['game_id'] == game]
            year = complete_path[-11:-7]
            eve_files = os.listdir(f'../data/game_data/{year}eve')
            for file in eve_files:
                if f'{complete_path[-11: -3]}EV' in file:
                    text_data = file
            
            #, ballpark_rating needs to be added again
            date, wind_factor, temp, hr_factor, ba_factor = self.stadium_info(f'../data/game_data/{year}eve/{text_data}', game) 
            
            for batter in current_game['batter'].unique():

                # Convert name to that used by Baseball-Reference
                bbref_name = playerid_reverse_lookup([batter], key_type='retro')
                bbref_name = bbref_name['key_bbref']
                n_index = str(bbref_name).index('\n')
                bbref_name = str(bbref_name)[5:n_index]

                

                current_batter = current_game[current_game['batter'] == batter]
                singles = current_batter[current_batter['result'] == 20].shape[0]
                doubles = current_batter[current_batter['result'] == 21].shape[0]
                triples = current_batter[current_batter['result'] == 22].shape[0]
                hrs = current_batter[current_batter['result'] == 23].shape[0] 
                rbis = current_batter['rbi'].sum()
                runs = self.calc_runs(current_game, batter, hrs)
                walks = current_batter[current_batter['result'] == 14].shape[0]
                sb = self.calc_sb(current_game, batter)
                hbp = current_batter[current_batter['result'] == 16].shape[0]
                ab = current_batter[current_batter['ab'] == 'T'].shape[0]
                starting_pitch_rpip = self.get_pitcher_era(current_batter.iloc[0]['pitcher'])
                

                games = self.batter_df[self.batter_df['playerID'] == bbref_name].shape[0]
                total_fantasy_points = self.batter_df[self.batter_df['playerID'] == bbref_name].sum()['fantasy_points']
                if games > 0:
                    points_per_game = total_fantasy_points/games
                else:
                    points_per_game = 0

                starting_pitcher_hand = current_batter.iloc[0, 4]
                total_vs_pitch_hand = self.batter_df[(self.batter_df['playerID'] == bbref_name) & (self.batter_df['starting_pitcher_hand'] == starting_pitcher_hand)].sum()['fantasy_points']
                games_vs_pitch_hand = self.batter_df[(self.batter_df['playerID'] == bbref_name) & (self.batter_df['starting_pitcher_hand'] == starting_pitcher_hand)].shape[0]

        
                if games_vs_pitch_hand > 0:
                    ppg_vs_hand = total_vs_pitch_hand / games_vs_pitch_hand
                else:
                    ppg_vs_hand = points_per_game

                

                fantasy_points = 3*singles + 6*doubles + 9*triples + 12*hrs + 3.5*rbis + 3.2*runs + 3*walks + 6*sb + 3*hbp


                # Add career numbers to the fantasy data
                career_nums = self.career_totals[self.career_totals['playerID'] == str(bbref_name)]
                career_nums['fantasy_ppg'] = float(points_per_game)
                career_nums['ppg_vs_hand'] = float(ppg_vs_hand)
                career_nums['starting_pitcher_hand'] = starting_pitcher_hand
                career_nums['fantasy_points'] = fantasy_points
                career_nums['wind_factor'] = wind_factor
                career_nums['temp'] = temp
                career_nums['starting_p_rpip'] = starting_pitch_rpip
                career_nums['date'] = date
                career_nums['season'] = year
                career_nums['position'] = current_batter.iloc[0]['batter_pos']
                career_nums['hr_factor'] = hr_factor
                career_nums['ba_factor'] = ba_factor

                # All of these features would be considered data leakage and cannot be used for modelling
                career_nums['singles_'] = singles
                career_nums['doubles_'] = doubles
                career_nums['triples_'] = triples
                career_nums['home_runs_'] = hrs
                career_nums['rbis_'] = rbis
                career_nums['runs_'] = runs
                career_nums['walks_'] = walks
                career_nums['stolen_bases_'] = sb
                career_nums['hbp_'] = hbp
                career_nums['ab_'] = ab

                seasonal_nums = self.batter_df[(self.batter_df['playerID'] == bbref_name) & (self.batter_df['season'] == year)]
                season_games = seasonal_nums.shape[0]
                if season_games == 0:
                    season_games += 1
                career_nums['seasonal_singles_p_game'] = seasonal_nums['singles_'].sum() / season_games
                career_nums['seasonal_doubles_p_game'] = seasonal_nums['doubles_'].sum() / season_games
                career_nums['seasonal_triples_p_game'] = seasonal_nums['triples_'].sum() / season_games
                career_nums['seasonal_hr_p_game'] = seasonal_nums['home_runs_'].sum() / season_games
                career_nums['seasonal_rbis_p_game'] = seasonal_nums['rbis_'].sum() / season_games
                career_nums['seasonal_runs_p_game'] = seasonal_nums['runs_'].sum() / season_games
                career_nums['seasonal_walks_p_game'] = seasonal_nums['walks_'].sum() / season_games
                career_nums['seasonal_sb_p_game'] = seasonal_nums['stolen_bases_'].sum() / season_games
                career_nums['seasonal_hbp_p_game'] = seasonal_nums['hbp_'].sum() / season_games
                career_nums['seasonal_ab_p_game'] = seasonal_nums['ab_'].sum() / season_games

                if season_games >= 7:
                    last_7 = seasonal_nums[-7:]
                    career_nums['last_7_fantasy'] = last_7['fantasy_points'].mean()
                else:
                    career_nums['last_7_fantasy'] = points_per_game
                self.batter_df = self.batter_df.append(career_nums)


    def pitching_career_totals(self):
        pitching = pitch()
        pitching = pitching.groupby('playerID').sum()
        pitching['runs_p_ip'] = pitching['R'] / (pitching['IPouts'] / 3)
        pitching = pitching.reset_index()
        return pitching


    def get_pitcher_era(self, pitcher_name):
        bbref_name = playerid_reverse_lookup([pitcher_name], key_type='retro')
        bbref_name = bbref_name['key_bbref']
        n_index = str(bbref_name).index('\n')
        bbref_name = str(bbref_name)[5:n_index]

        pitcher_stats = self.pitcher_df[self.pitcher_df['playerID'] == bbref_name]
        
        return pitcher_stats['runs_p_ip'].mean()
        


    def calc_runs(self, game_df, batter, hrs):
        # Helper function to return the number of runs a player scored in a game 
        runs = hrs
        runs += game_df[(game_df['run_first'] == batter) & (game_df['first_dest'].isin([4,5,6]))].shape[0]
        runs += game_df[(game_df['run_second'] == batter) & (game_df['second_dest'].isin([4,5,6]))].shape[0]
        runs += game_df[(game_df['run_third'] == batter) & (game_df['third_dest'].isin([4,5,6]))].shape[0]
        return runs

    
    def calc_sb(self, game_df, batter):
        # Helper function to return the number of stolen bases a player had in a game 
        sb = 0 
        sb += game_df[(game_df['run_first'] == batter) & (game_df['sb_first'] == 'T')].shape[0]
        sb += game_df[(game_df['run_second'] == batter) & (game_df['sb_second'] == 'T')].shape[0]
        sb += game_df[(game_df['run_third'] == batter) & (game_df['sb_third'] == 'T')].shape[0]
        return sb


    def stadium_info(self, event_file_path, game_id):
        # Helper function that returns information about the stadium being played in 
        f = open(event_file_path)
        text = f.read()
        game_idx = text.index(game_id)
        
        date_idx = text.index('date', game_idx)
        date = text[date_idx+5:date_idx+15]
        date = pd.to_datetime(date)
        
        winddir_idx = text.index('winddir,', game_idx)
        winddir = text[winddir_idx+8:winddir_idx+12]
        if winddir in ['ltor', 'rtol', 'unkn']:
            winddir = 0
        elif winddir in ['tocf', 'torf', 'tolf']:
            winddir = 1
        else:
            winddir = -1
        
        windspeed_idx = text.index('windspeed', game_idx)
        windspeed = text[windspeed_idx+10:windspeed_idx+11]
        wind_factor = winddir * int(windspeed)
        
        temp_idx = text.index('info,temp,', game_idx)
        temp = int(text[temp_idx+10:temp_idx+12])

        stadium = event_file_path[-7:-4]
        if stadium in self.ballpark_factors_hr.keys():
            hr_factor = self.ballpark_factors_hr[stadium]
            ba_factor = self.ballpark_factors_ba[stadium]
        else:
            print(stadium)
            hr_factor = 1
            ba_factor = 1
        f.close()
    
        return date, wind_factor, temp, hr_factor, ba_factor

    def save_dataframe(self, file_path):
        self.batter_df.to_csv(file_path)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    constructor = Constructor('../data/processed_events')
    constructor.create_training_dataframe()
    constructor.save_dataframe('../data/completed_df_part2.csv')

    
