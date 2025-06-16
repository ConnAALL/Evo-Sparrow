#!/usr/bin/env python3
"""
Agent Ranking Analysis Script

This script analyzes the performance of different agents in game matches
and provides multiple mathematical approaches to determine the strongest agent.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import math


class AgentRankingAnalyzer:
    def __init__(self, csv_file):
        """Initialize the analyzer with the CSV file."""
        self.df = pd.read_csv(csv_file)
        self.matches = self._parse_matches()
        
    def _parse_matches(self):
        """Parse the CSV into individual matches."""
        matches = []
        excluded_matches = 0
        
        # Remove rows with all NaN values (empty rows)
        df_clean = self.df.dropna(how='all')
        
        # Group every 3 rows as a match
        for i in range(0, len(df_clean), 3):
            if i + 2 < len(df_clean):
                match = df_clean.iloc[i:i+3].copy()
                # Only include matches with complete data (no missing scores or agent names)
                if (not match['Avg. Score'].isna().any() and 
                    not match['Game'].isna().any() and 
                    len(match) == 3 and
                    all(isinstance(score, (int, float)) for score in match['Avg. Score'])):
                    matches.append(match)
                else:
                    excluded_matches += 1
                    print(f"Excluding incomplete match {excluded_matches}: {match['Game'].values}")
        
        print(f"Total complete matches found: {len(matches)}")
        print(f"Incomplete matches excluded: {excluded_matches}")
        
        return matches
    
    def get_head_to_head_matrix(self):
        """Create a head-to-head win matrix."""
        agents = set()
        for match in self.matches:
            agents.update(match['Game'].values)
        
        agents = sorted(list(agents))
        n_agents = len(agents)
        
        # Initialize matrices
        win_matrix = defaultdict(lambda: defaultdict(int))
        total_games = defaultdict(lambda: defaultdict(int))
        
        for match in self.matches:
            # Sort by average score to determine ranking in this match
            match_sorted = match.sort_values('Avg. Score', ascending=False)
            agents_in_match = match_sorted['Game'].values
            scores = match_sorted['Avg. Score'].values
            
            # Count wins (who beats whom)
            for i, agent1 in enumerate(agents_in_match):
                for j, agent2 in enumerate(agents_in_match):
                    if i != j:
                        total_games[agent1][agent2] += 1
                        if scores[i] > scores[j]:
                            win_matrix[agent1][agent2] += 1
        
        return win_matrix, total_games, agents
    
    def calculate_win_percentages(self):
        """Calculate win percentages for each agent against others."""
        win_matrix, total_games, agents = self.get_head_to_head_matrix()
        
        win_percentages = {}
        for agent1 in agents:
            win_percentages[agent1] = {}
            for agent2 in agents:
                if agent1 != agent2 and total_games[agent1][agent2] > 0:
                    win_pct = win_matrix[agent1][agent2] / total_games[agent1][agent2]
                    win_percentages[agent1][agent2] = win_pct
                else:
                    win_percentages[agent1][agent2] = 0.0
        
        return win_percentages, agents
    
    def calculate_elo_ratings(self, k_factor=32, initial_rating=1500):
        """Calculate Elo ratings based on match results."""
        win_matrix, total_games, agents = self.get_head_to_head_matrix()
        
        # Initialize Elo ratings
        elo_ratings = {agent: initial_rating for agent in agents}
        
        # Process each match
        for match in self.matches:
            match_sorted = match.sort_values('Avg. Score', ascending=False)
            agents_in_match = match_sorted['Game'].values
            scores = match_sorted['Avg. Score'].values
            
            # Update Elo for each pair in the match
            for i, agent1 in enumerate(agents_in_match):
                for j, agent2 in enumerate(agents_in_match):
                    if i != j:
                        # Expected score for agent1 against agent2
                        expected_score = 1 / (1 + 10**((elo_ratings[agent2] - elo_ratings[agent1]) / 400))
                        
                        # Actual score (1 if agent1 wins, 0 if loses)
                        actual_score = 1 if scores[i] > scores[j] else 0
                        
                        # Update Elo rating
                        elo_ratings[agent1] += k_factor * (actual_score - expected_score)
        
        return elo_ratings
    
    def calculate_tournament_points(self):
        """Calculate tournament-style points (3 for 1st, 2 for 2nd, 1 for 3rd, etc.)."""
        agent_points = defaultdict(int)
        agent_matches = defaultdict(int)
        
        for match in self.matches:
            match_sorted = match.sort_values('Avg. Score', ascending=False)
            agents_in_match = match_sorted['Game'].values
            
            # Award points based on ranking
            points = [3, 2, 1]  # Assuming 3 agents per match
            for i, agent in enumerate(agents_in_match):
                if i < len(points):
                    agent_points[agent] += points[i]
                agent_matches[agent] += 1
        
        # Calculate average points per match
        avg_points = {agent: agent_points[agent] / agent_matches[agent] 
                     for agent in agent_points if agent_matches[agent] > 0}
        
        return avg_points, agent_points, agent_matches
    
    def calculate_bradley_terry_model(self, max_iterations=1000, tolerance=1e-6):
        """Calculate Bradley-Terry model strengths."""
        win_matrix, total_games, agents = self.get_head_to_head_matrix()
        n_agents = len(agents)
        
        # Initialize strengths uniformly
        strengths = {agent: 1.0 for agent in agents}
        
        for iteration in range(max_iterations):
            new_strengths = {}
            
            for agent in agents:
                numerator = sum(win_matrix[agent][opponent] for opponent in agents if opponent != agent)
                denominator = sum(total_games[agent][opponent] * strengths[agent] / 
                                (strengths[agent] + strengths[opponent]) 
                                for opponent in agents if opponent != agent and total_games[agent][opponent] > 0)
                
                if denominator > 0:
                    new_strengths[agent] = numerator / denominator
                else:
                    new_strengths[agent] = strengths[agent]
            
            # Normalize to prevent explosion
            total_strength = sum(new_strengths.values())
            if total_strength > 0:
                new_strengths = {agent: strength / total_strength * n_agents 
                               for agent, strength in new_strengths.items()}
            
            # Check for convergence
            converged = all(abs(new_strengths[agent] - strengths[agent]) < tolerance 
                          for agent in agents)
            
            strengths = new_strengths
            
            if converged:
                break
        
        return strengths
    
    def print_comprehensive_analysis(self):
        """Print comprehensive analysis of all ranking methods."""
        print("=" * 80)
        print("COMPREHENSIVE AGENT RANKING ANALYSIS")
        print("=" * 80)
        
        print(f"\nTotal matches analyzed: {len(self.matches)}")
        
        # 1. Head-to-head win percentages
        print("\n" + "1. HEAD-TO-HEAD WIN PERCENTAGES")
        print("-" * 50)
        win_percentages, agents = self.calculate_win_percentages()
        
        print(f"{'Agent':<8}", end="")
        for agent in agents:
            print(f"{agent:>8}", end="")
        print()
        
        for agent1 in agents:
            print(f"{agent1:<8}", end="")
            for agent2 in agents:
                if agent1 == agent2:
                    print(f"{'--':>8}", end="")
                else:
                    pct = win_percentages[agent1][agent2] * 100
                    print(f"{pct:>7.1f}%", end="")
            print()
        
        # 2. Overall win rate
        print("\n" + "2. OVERALL WIN RATE AGAINST ALL OPPONENTS")
        print("-" * 50)
        overall_win_rates = {}
        for agent in agents:
            total_wins = sum(win_percentages[agent][opp] * 
                           len([m for m in self.matches if agent in m['Game'].values and opp in m['Game'].values])
                           for opp in agents if opp != agent)
            total_games = sum(len([m for m in self.matches if agent in m['Game'].values and opp in m['Game'].values])
                            for opp in agents if opp != agent)
            overall_win_rates[agent] = (total_wins / total_games * 100) if total_games > 0 else 0
        
        for agent, rate in sorted(overall_win_rates.items(), key=lambda x: x[1], reverse=True):
            print(f"{agent}: {rate:.1f}%")
        
        # 3. Elo ratings
        print("\n" + "3. ELO RATINGS")
        print("-" * 50)
        elo_ratings = self.calculate_elo_ratings()
        for agent, rating in sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True):
            print(f"{agent}: {rating:.0f}")
        
        # 4. Tournament points
        print("\n" + "4. TOURNAMENT POINTS SYSTEM")
        print("-" * 50)
        avg_points, total_points, total_matches = self.calculate_tournament_points()
        print(f"{'Agent':<8} {'Avg Points':<12} {'Total Points':<13} {'Matches':<8}")
        print("-" * 45)
        for agent in sorted(avg_points.keys(), key=lambda x: avg_points[x], reverse=True):
            print(f"{agent:<8} {avg_points[agent]:<12.2f} {total_points[agent]:<13} {total_matches[agent]:<8}")
        
        # 5. Bradley-Terry model
        print("\n" + "5. BRADLEY-TERRY MODEL STRENGTHS")
        print("-" * 50)
        bt_strengths = self.calculate_bradley_terry_model()
        for agent, strength in sorted(bt_strengths.items(), key=lambda x: x[1], reverse=True):
            print(f"{agent}: {strength:.3f}")
        
        # 6. Final ranking consensus
        print("\n" + "6. CONSENSUS RANKING")
        print("-" * 50)
        rankings = {}
        
        # Rank by each method
        methods = {
            'overall_win_rate': overall_win_rates,
            'elo': elo_ratings,
            'tournament_points': avg_points,
            'bradley_terry': bt_strengths
        }
        
        for method, scores in methods.items():
            sorted_agents = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
            for i, agent in enumerate(sorted_agents):
                if agent not in rankings:
                    rankings[agent] = []
                rankings[agent].append(i + 1)
        
        # Calculate average ranking
        avg_rankings = {agent: np.mean(ranks) for agent, ranks in rankings.items()}
        
        print(f"{'Agent':<8} {'Avg Rank':<10} {'Rankings (Win%, Elo, Tourn, B-T)'}")
        print("-" * 55)
        for agent in sorted(avg_rankings.keys(), key=lambda x: avg_rankings[x]):
            ranks_str = ", ".join(map(str, rankings[agent]))
            print(f"{agent:<8} {avg_rankings[agent]:<10.1f} [{ranks_str}]")
        
        # Determine strongest agent
        strongest_agent = min(avg_rankings.keys(), key=lambda x: avg_rankings[x])
        print(f"\n🏆 STRONGEST AGENT: {strongest_agent}")
        print(f"   Average ranking across all methods: {avg_rankings[strongest_agent]:.1f}")
        
        return strongest_agent, avg_rankings


def main():
    """Main function to run the analysis."""
    try:
        analyzer = AgentRankingAnalyzer('results.csv')
        strongest_agent, rankings = analyzer.print_comprehensive_analysis()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"The mathematically strongest agent is: {strongest_agent}")
        
    except FileNotFoundError:
        print("Error: results.csv file not found in the current directory.")
        print("Please make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error occurred during analysis: {str(e)}")


if __name__ == "__main__":
    main()