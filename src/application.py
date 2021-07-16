from datetime import date, datetime
from tkinter import *
from numpy import testing
import pandas as pd
from pybaseball import batting_stats, teams
from pybaseball.lahman import *
from PIL import ImageTk, Image
from bs4 import BeautifulSoup
import requests
from pybaseball import playerid_reverse_lookup
import warnings
import joblib
import numpy as np
from sklearn.impute import SimpleImputer


class Predictor:

    def __init__(self):
        self.career_nums = self.career_numbers()
        self.df_flag = False
        self.daily_df = None
        self.root = Tk()
        self.teams = {'ANA':'Angels',
                        'BAL':'Orioles',
                        'BOS': 'Red Sox',
                        'CHW':'White Sox',
                        'CLE':'Indians',
                        'DET':'Tigers',
                        'HOU':'Astros',
                        'KAN': 'Royals', 
                        'MIN': 'Twins',
                        'NYY':'Yankees',
                        'OAK':'Athletics',
                        'SEA':'Mariners',
                        'TAM': 'Rays',
                        'TEX':'Rangers',
                        'TOR': 'Blue Jays',
                        'ARI': 'Diamondbacks',
                        'ATL': 'Braves',
                        'CHC':'Cubs',
                        'CIN':'Reds',
                        'COL': 'Rockies',
                        'FLA':'Marlins',
                        'LOS': 'Dodgers',
                        'MIL':'Brewers',
                        'NYM': 'Mets',
                        'PHI': 'Phillies',
                        'PIT':'Pirates',
                        'SDG': 'Padres',
                        'SFO':'Giants',
                        "STL":'Cardinals',
                        'WAS':'Nationals'}
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
                                    'STL': .87,
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
                                    'STL': .99,
                                    'TBA': .95,
                                    'TEX': .99,
                                    'TOR': 1.02,
                                    'WAS': .99}
        self.ppg_v_lhp = None
        self.ppg_v_rhp = None
        self.main_screen()
        

    def career_numbers(self):
        download_lahman()
        batting_numbers = batting()
        fielding_nums = fielding()
        fielding_nums = fielding_nums.loc[:,['playerID', 'POS']]
        recent_years = batting_numbers[batting_numbers['yearID'] > 1980]

        # Limit to players who have played since 1980
        career_totals = recent_years.groupby('playerID').sum()
        career_totals.reset_index()
        career_totals = pd.merge(career_totals, fielding_nums, how='left', on='playerID')
        
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
       
        career_totals = career_totals.iloc[:,[0, 20,21,22,23,24,25,26,27,28,29]]
        career_totals.loc[career_totals['POS'] == 'P', 'POS_num'] = 1
        
        career_totals.loc[career_totals['POS'] == 'C', 'POS_num'] = 2
        career_totals.loc[career_totals['POS'] == '1B', 'POS_num'] = 3
        career_totals.loc[career_totals['POS'] == '2B', 'POS_num'] = 4
        career_totals.loc[career_totals['POS'] == '3B', 'POS_num'] = 5
        career_totals.loc[career_totals['POS'] == 'SS', 'POS_num'] = 6
        career_totals.loc[career_totals['POS'] == 'OF', 'POS_num'] = 7
        career_totals = career_totals.groupby('playerID').mean()
        career_totals['POS_num'] = career_totals['POS_num'].astype(int)
        length = career_totals.loc[career_totals['POS_num'] == 7, 'POS_num'].shape[0]
        career_totals.loc[career_totals['POS_num'] == 7, 'POS_num'] = np.random.randint(7, 10, length)

        career_totals = pd.get_dummies(career_totals, columns=['POS_num'])
        career_totals['POS_num_10'] = 0
        career_totals['POS_num_11'] = 0







        career_totals = career_totals.reset_index()

        return career_totals
       

    def main_screen(self):
        self.root.configure(bg='gray11')
        self.root.title('Machine Learning Baseball')
        
        logo = Canvas(self.root, width = 450, height = 350, bg='gray11', highlightthickness=0)
        logo.grid(row = 0, column=0)
        img = ImageTk.PhotoImage(Image.open('../images/machine_learning_baseball.jpg'))
        logo.create_image(20,20,  anchor=NW,image=img)

        input_label = Label(self.root, text='ENTER A DATE BELOW TO PREDICT FANTASY STATS',bg='gray11', fg='white')
        input_label.grid(row=1, column=0)
        date_entry = Entry(self.root, width=15)
        date_entry.grid(row=2, column=0)
        
        go_button = Button(self.root, text='Make Prediction', command=lambda:self.prediction(date_entry.get()))
        go_button.grid(row=3, column=0)



        self.root.mainloop()

    def prediction(self, date):
        date = pd.to_datetime(date)
        self.get_data(date)
        testing_vals = self.daily_df.iloc[:,[35, 36,37,38,39,40,41,42,43,14,32,29,30,27, 25,26,15,16,17,18,19,20,24,21,22,23,55,44,45,46,47,48,49,50,51,52,53,54]]
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        testing_vals = imp.fit_transform(testing_vals)
        # Load in trained models
        model_1 = joblib.load('fit_models/1_xgboost.joblib')
        model_2 = joblib.load('fit_models/2_xgboost.joblib')
        model_3 = joblib.load('fit_models/3_line.joblib')
        y_hat_1 = model_1.predict(testing_vals)
        y_hat_2 = model_2.predict(testing_vals)
        y_hat_3 = model_3.predict(testing_vals)

        file = open("fit_models/weights.txt", "r")
        weights = eval(file.read())

        total_y_hat = (y_hat_1 * weights['w1'] + y_hat_2 * weights['w2'] + y_hat_3 * weights['w3']) / (weights['w1'] + weights['w2'] + weights['w3'])
        self.daily_df['predicted_points'] =  total_y_hat
        self.daily_df = self.daily_df.sort_values('predicted_points', ascending=False)
        print(self.daily_df.iloc[:,[1, 56]])
        self.daily_df.iloc[:,[1, 56]].to_csv('../my_first_prediction.csv')


        




    def get_data(self, date):
        self.populate_pitcher_splits(date)
        current_schedule = pd.read_excel(f'http://dailybaseballdata.com/dbd/MLB_schedule_{date.year}.xls', skiprows=1)
        current_schedule = current_schedule.rename(columns={int(date.year): 'date'})
        games_on_date = current_schedule[current_schedule['date'] == f'{date.year}-{date.month}-{date.day}']
        games_on_date = games_on_date.dropna(axis=1)
        for col in games_on_date.copy().columns:
            if '@' in str(games_on_date.loc[:, col]):

                games_on_date = games_on_date.drop(columns=col)
        
        current_season = batting_stats(start_season=date.year, qual=20)
        current_season = current_season.iloc[:, [0,2,3,5,9,10,11,12,13,14,15,18,22,6]]
        current_season['fantasy_ppg'] = (3*current_season['1B'] + 6*current_season['2B'] + 9*current_season['3B'] + 12*current_season['HR'] + 3.5*current_season['RBI'] + 3.2*current_season['R'] + 3*current_season['BB'] + 6*current_season['SB'] + 3*current_season['HBP']) / current_season['G']
        current_season[current_season['G'] == 0]['G'] = 1
        current_season['seasonal_singles_p_game'] = current_season['1B'] / current_season['G']
        current_season['seasonal_doubles_p_game'] = current_season['2B'] / current_season['G']
        current_season['seasonal_triples_p_game'] = current_season['3B'] / current_season['G']
        current_season['seasonal_hr_p_game'] = current_season['HR'] / current_season['G']
        current_season['seasonal_rbis_p_game'] = current_season['RBI'] / current_season['G']
        current_season['seasonal_runs_p_game'] = current_season['R'] / current_season['G']
        current_season['seasonal_sb_p_game'] = current_season['SB'] / current_season['G']
        current_season['seasonal_hbp_p_game'] = current_season['HBP'] / current_season['G']
        current_season['seasonal_ab_p_game'] = current_season['AB'] / current_season['G']
        current_season['seasonal_bb_p_game'] = current_season['BB'] / current_season['G']

        for i, col in enumerate(games_on_date.columns[2:]):
            home_team = col.upper()
            away_team = games_on_date.iloc[0,i+2].upper()
            players_home = current_season[current_season['Team'] == home_team]
            players_away = current_season[current_season['Team'] == away_team]
            

            # Setting ballpark factor
            if home_team in self.ballpark_factors_hr.keys():
                ballpark_factor_hr = self.ballpark_factors_hr[home_team]
                ballpark_factor_ba = self.ballpark_factors_ba[home_team]
            else:
                ballpark_factor_hr = 1
                ballpark_factor_ba = 1
            players_away['ballpark_factor_hr'] = ballpark_factor_hr
            players_away['ballpark_factor_ba'] = ballpark_factor_ba

            players_home['ballpark_factor_hr'] = ballpark_factor_hr
            players_home['ballpark_factor_ba'] = ballpark_factor_ba
            try:
                temp, wind_factor= self.scrape_weather(home_team, date)
                print(f'Actually got the weather for game at {home_team}')
            except:
                temp, wind_factor = 72, 0
                print(f'Could not get weather for game at {home_team}')

            
            try:
                away_pitch_hand, away_rpip, home_pitch_hand, home_rpip = self.scrape_prob_pitcher(home_team, away_team, date)
            except:
                away_pitch_hand, away_rpip, home_pitch_hand, home_rpip = 'R', .493, 'R', '.493'
                print('Could not collect opposing pitcher data')
            players_away['starting_pitch_rpip'] = home_rpip
            players_away['pitcher_hand'] = home_pitch_hand

            players_home['starting_pitch_rpip'] = away_rpip
            players_home['pitcher_hand'] = away_pitch_hand
            players_home['wind_factor'] = wind_factor
            players_away['wind_factor'] = wind_factor
            players_home['temp'] = temp
            players_away['temp'] = temp

            if away_pitch_hand == 'L':
                players_home = players_home.merge(self.ppg_v_lhp, how='left', left_on='IDfg', right_on='playerId')
            else:
                players_home = players_home.merge(self.ppg_v_rhp, how='left', left_on='IDfg', right_on='playerId')

            if home_pitch_hand == 'L':
                players_away = players_away.merge(self.ppg_v_lhp, how='left', left_on='IDfg', right_on='playerId')
            else:
                players_away = players_away.merge(self.ppg_v_rhp, how='left', left_on='IDfg', right_on='playerId')
            
            player_ids_home = playerid_reverse_lookup(players_home['IDfg'].tolist(), key_type='fangraphs')['key_bbref']
            players_home['bbref'] = player_ids_home

            player_ids_away = playerid_reverse_lookup(players_away['IDfg'].tolist(), key_type='fangraphs')['key_bbref']
            players_away['bbref'] = player_ids_away

            players_home = players_home.merge(self.career_nums, how='left', left_on='bbref', right_on='playerID')
            players_away = players_away.merge(self.career_nums, how='left', left_on='bbref', right_on='playerID')
            print(away_team)
            print(players_away)
            print(home_team)
            print(players_home)
           
            if self.df_flag == False:
                self.daily_df = players_home
                self.daily_df = self.daily_df.append(players_away)
            else:
                self.daily_df = self.daily_df.append(players_home)
                self.daily_df = self.daily_df.append(players_away)
            self.df_flag = True
        self.daily_df['last_7_fantasy'] = self.daily_df['fantasy_ppg']
                
        for i, col in enumerate(self.daily_df):
            print(col, i)    
            
            

            
    def populate_pitcher_splits(self, date):
        vs_rhp = pd.read_csv('../data/Splits Leaderboard Data vs RHP.csv')
        vs_lhp = pd.read_csv('../data/Splits Leaderboard Data vs LHP.csv')
        vs_rhp['ppg_vs_hand'] = (vs_rhp['1B'] * 3 + vs_rhp['2B'] * 6 + vs_rhp['3B'] * 9 + vs_rhp['HR'] * 12 + vs_rhp['RBI'] * 3.5 + vs_rhp['R'] * 3.2 + vs_rhp['BB'] * 3 + vs_rhp['SB'] * 6 + vs_rhp['HBP'] * 3) / vs_rhp['G']
        vs_lhp['ppg_vs_hand'] = (vs_lhp['1B'] * 3 + vs_lhp['2B'] * 6 + vs_lhp['3B'] * 9 + vs_lhp['HR'] * 12 + vs_lhp['RBI'] * 3.5 + vs_lhp['R'] * 3.2 + vs_lhp['BB'] * 3 + vs_lhp['SB'] * 6 + vs_lhp['HBP'] * 3) / vs_lhp['G']
        self.ppg_v_rhp = vs_rhp[['playerId', 'ppg_vs_hand']]
        self.ppg_v_lhp = vs_lhp[['playerId', 'ppg_vs_hand']]


    def scrape_weather(self, home_team, date):
        if len(str(date.month)) == 1:
            month = f'0{date.month}'
        else:
            month = date.month
        if len(str(date.day)) == 1:
            day = f'0{date.day}'
        else:
            day = date.day    
        page = requests.get(f'https://swishanalytics.com/mlb/weather?date={date.year}-{month}-{day}')
        soup = BeautifulSoup(page.content, "html.parser")
        games = soup.find_all('div', class_='weather-card')
        for game in games:
            teams = game.find('h4').text.strip()
            home = teams[teams.index('@')+3:]
            home_mascot = self.teams[home_team]
            if home == home_mascot:
                temp = int(game.find(class_='text-center gametime-hour').text[:-1])
                rows = game.find_all(class_='text-center gametime-hour')
                wind_inx = str(rows[9]).find('>')
                wind = float(str(rows[9])[wind_inx+1:wind_inx+5])
                images = game.find_all(class_='mar-0')
            
                r_index = str(images[13]).find('rotate')
               
                wind_dir = int(float(str(images[13])[(r_index + 7):(r_index+12)]))
                if wind_dir < 0:
                    wind_dir = 360 + wind_dir

                if wind_dir < 60 and wind_dir >=0:
                    wind_dir = 1
                elif wind_dir >= 60 and wind_dir < 120:
                    wind_dir = 0
                elif wind_dir >= 120 and wind_dir < 240:
                    wind_dir = -1
                elif wind_dir >= 240 and wind_dir < 300:
                    wind_dir = 0
                else:
                    wind_dir = 1
                wind_factor = wind_dir*wind


                
                return temp, wind_factor



        

    
    def scrape_prob_pitcher(self, home_team, away_team, date):
        if len(str(date.month)) == 1:
            month = f'0{date.month}'
        else:
            month = date.month
        if len(str(date.day)) == 1:
            day = f'0{date.day}'
        else:
            day = date.day          
        mod_date = f'{date.year}{month}{day}'
        url = f'https://www.cbssports.com/fantasy/baseball/probable-pitchers/{mod_date}/'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        games = soup.find_all(class_='ProbablePitchersTableCard')
        away_pitch_hand = 'R'
        home_pitch_hand = 'L'
        away_rpip = .493
        home_rpip = .493
        for game in games:
            away = game.find_next(class_="CellPlayerName-team").text
            if away.strip() == away_team:
                hands = game.find_all('span', class_="CellPlayerName-handPreference")
                away_pitch_hand = hands[1].text.strip()
                if away_pitch_hand == 'RHP':
                    away_pitch_hand = 'R'
                else:
                    away_pitch_hand = 'L'

                home_pitch_hand = hands[2].text.strip()
                if home_pitch_hand == 'RHP':
                    home_pitch_hand = 'R'
                else:
                    home_pitch_hand = 'L'
                try:
                    away_rpip = float(game.find_all(class_='TableBase-bodyTd')[3].text.strip()) / 9
                    home_rpip = float(game.find_all(class_='TableBase-bodyTd')[12].text.strip()) / 9
                except:
                    pass
        return away_pitch_hand, away_rpip, home_pitch_hand, home_rpip

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    app = Predictor()
    

