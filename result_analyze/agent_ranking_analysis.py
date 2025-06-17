#!/usr/bin/env python3
"""
Agent Ranking Analysis Script
  • handles “%” columns
  • supports N-player groupings
  • head-to-head, overall win-rate
  • multiplayer Elo
  • tournament points
  • Bradley–Terry
  • raw W/D/L/Deal-In stats
  • consensus ranking
"""

import pandas as pd
import numpy as np
from collections import defaultdict

class AgentRankingAnalyzer:
    def __init__(self, csv_file, group_size=3, k_factor=32, initial_elo=1500):
        # --- load & clean ---
        self.df = pd.read_csv(csv_file)
        # strip “%” from these columns and convert to [0..1]
        for col in ('Wins','Draws','Losses','Deal-Ins'):
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col].astype(str)
                              .str.rstrip('%')
                              .astype(float)
                              .div(100)
                )
        self.group_size = group_size
        self.k = k_factor
        self.init_elo = initial_elo
        self.matches = self._parse_matches()
        self.agents = sorted(self.df['Game'].unique())

    def _parse_matches(self):
        """Group every group_size rows as one free-for-all match."""
        df = self.df.dropna(how='all').reset_index(drop=True)
        matches = []
        for i in range(0, len(df), self.group_size):
            block = df.iloc[i:i+self.group_size]
            if len(block) == self.group_size and not block['Avg. Score'].isna().any():
                matches.append(block.copy())
        print(f"Parsed {len(matches)} complete matches (group_size={self.group_size})")
        return matches

    # ── HEAD-TO-HEAD & OVERALL WIN-RATES ─────────────────────────────────────────

    def get_head_to_head_matrix(self):
        """Return win_matrix[A][B] and total_games[A][B]."""
        win = defaultdict(lambda: defaultdict(int))
        total = defaultdict(lambda: defaultdict(int))
        for m in self.matches:
            sorted_m = m.sort_values('Avg. Score', ascending=False)
            names = sorted_m['Game'].values
            scores = sorted_m['Avg. Score'].values
            for i, A in enumerate(names):
                for j, B in enumerate(names):
                    if A==B: continue
                    total[A][B] += 1
                    if scores[i] > scores[j]:
                        win[A][B] += 1
        return win, total

    def calculate_win_percentages(self):
        win, total = self.get_head_to_head_matrix()
        wpct = {A:{} for A in self.agents}
        for A in self.agents:
            for B in self.agents:
                if A==B or total[A][B]==0:
                    wpct[A][B] = 0.0
                else:
                    wpct[A][B] = win[A][B]/total[A][B]
        return wpct

    def calculate_overall_win_rate(self, win_pct):
        """Weighted average of pairwise win% by encounter count."""
        overall = {}
        for A in self.agents:
            num, den = 0.0, 0
            for B in self.agents:
                if A==B: continue
                # count matches where A and B both appear
                cnt = sum(1 for m in self.matches
                          if A in m['Game'].values and B in m['Game'].values)
                num += win_pct[A][B] * cnt
                den += cnt
            overall[A] = (num/den if den>0 else 0.0)
        return overall

    # ── MULTIPLAYER ELO ──────────────────────────────────────────────────────────

    def calculate_multiplayer_elo(self):
        R = {a: self.init_elo for a in self.agents}

        for m in self.matches:
            sorted_m = m.sort_values('Avg. Score', ascending=False)
            # only unique agent names, in rank order
            players = []
            for g in sorted_m['Game']:
                if g not in players:
                    players.append(g)

            # skip if there’s nobody to play against
            if len(players) < 2:
                continue

            N = len(players)
            # actual scores 1.0→0.0 for first→last
            S = { players[i]: (N-1 - i)/(N-1) for i in range(N) }

            # expected = average pairwise expectation
            E = {}
            for A in players:
                exps = []
                for B in players:
                    if A == B: 
                        continue
                    exps.append(1 / (1 + 10**((R[B] - R[A]) / 400)))
                E[A] = sum(exps) / len(exps)

            # update
            for A in players:
                R[A] += self.k * (S[A] - E[A])

        return R

    # ── TOURNAMENT POINTS ────────────────────────────────────────────────────────

    def calculate_tournament_points(self):
        pts = defaultdict(int)
        cnt = defaultdict(int)
        for m in self.matches:
            sorted_m = m.sort_values('Avg. Score', ascending=False)
            names = list(sorted_m['Game'])
            # e.g. for 3 players use [3,2,1], for 4 use [4,3,2,1]
            points = list(range(self.group_size, 0, -1))
            for i,A in enumerate(names):
                pts[A] += points[i]
                cnt[A] += 1
        avg_pts = {A: pts[A]/cnt[A] for A in pts}
        return avg_pts, pts, cnt

    # ── BRADLEY–TERRY ───────────────────────────────────────────────────────────

    def calculate_bradley_terry(self, max_iter=1000, tol=1e-6):
        win, total = self.get_head_to_head_matrix()
        # init strengths
        s = {A:1.0 for A in self.agents}
        for _ in range(max_iter):
            s_new = {}
            for A in self.agents:
                num = sum(win[A][B] for B in self.agents if B!=A)
                den = sum(total[A][B]*s[A]/(s[A]+s[B])
                          for B in self.agents
                          if B!=A and total[A][B]>0)
                s_new[A] = num/den if den>0 else s[A]
            # normalize
            tot = sum(s_new.values())
            s_new = {A: (s_new[A]/tot*len(self.agents)) for A in s_new}
            if all(abs(s_new[A]-s[A])<tol for A in self.agents):
                break
            s = s_new
        return s

    # ── RAW AVG W/D/L/Deal-In ────────────────────────────────────────────────────

    def calculate_raw_stats(self):
        """Mean of your per-row % columns (Wins/Draws/Losses/Deal-Ins)."""
        raw = {}
        for A, dfA in self.df.groupby('Game'):
            raw[A] = dfA[['Wins','Draws','Losses','Deal-Ins']].mean().to_dict()
        return raw

    # ── CONSENSUS RANKING ────────────────────────────────────────────────────────

    def calculate_consensus(self, methods: dict):
        """
        methods: name → {agent:score}
        lower rank=better, so we sort each method descending and record 1-based ranks.
        """
        ranks = defaultdict(list)
        for name, scores in methods.items():
            sorted_agents = sorted(scores, key=lambda A: scores[A], reverse=True)
            for idx,A in enumerate(sorted_agents, start=1):
                ranks[A].append(idx)
        # average rank
        avg_rank = {A: np.mean(ranks[A]) for A in ranks}
        return avg_rank, ranks

    # ── PRINT EVERYTHING ───────────────────────────────────────────────────────

    def print_comprehensive_analysis(self):
        print("\n" + "="*60)
        print(" COMPREHENSIVE AGENT RANKING ANALYSIS ")
        print("="*60)
        print(f"Total matches analyzed: {len(self.matches)}")

        # 1. head-to-head %
        print("\n1) HEAD-TO-HEAD WIN %")
        win_pct = self.calculate_win_percentages()
        agents = self.agents
        header = "Agent   " + "".join(f"{a:>8}" for a in agents)
        print(header)
        for A in agents:
            row = f"{A:<7}"
            for B in agents:
                row += f"{(win_pct[A][B]*100):8.1f}" if A!=B else f"{'   --':>8}"
            print(row, "%")

        # 2. overall win-rate
        print("\n2) OVERALL WIN RATE")
        overall = self.calculate_overall_win_rate(win_pct)
        for A,r in sorted(overall.items(), key=lambda x:-x[1]):
            print(f"  {A:<7} {r*100:5.1f}%")

        # 3. multiplayer Elo
        print("\n3) MULTIPLAYER ELO")
        elo = self.calculate_multiplayer_elo()
        for A,r in sorted(elo.items(), key=lambda x:-x[1]):
            print(f"  {A:<7} {r:7.1f}")

        # 4. tournament points
        print("\n4) TOURNAMENT POINTS")
        avg_pts, tot_pts, cnt = self.calculate_tournament_points()
        print("Agent    AvgPts   TotPts  Matches")
        for A in sorted(avg_pts, key=lambda x:-avg_pts[x]):
            print(f"  {A:<7} {avg_pts[A]:7.2f} {tot_pts[A]:8d} {cnt[A]:8d}")

        # 5. Bradley–Terry strengths
        print("\n5) BRADLEY–TERRY STRENGTHS")
        bt = self.calculate_bradley_terry()
        for A,s in sorted(bt.items(), key=lambda x:-x[1]):
            print(f"  {A:<7} {s:7.3f}")

        # 6. raw W/D/L/Deal-Ins
        print("\n6) RAW AVG W/D/L/DEAL-IN %")
        raw = self.calculate_raw_stats()
        for A, d in raw.items():
            print(
                f"  {A:<7} "
                f"W {d['Wins']*100:5.1f}%  "
                f"D {d['Draws']*100:5.1f}%  "
                f"L {d['Losses']*100:5.1f}%  "
                f"DI {d['Deal-Ins']*100:5.1f}%"
            )

        # 7. consensus ranking
        methods = {
            'Win%': {A: overall[A] for A in agents},
            'Elo': elo,
            'Tourn': avg_pts,
            'BT': bt
        }
        avg_rank, all_ranks = self.calculate_consensus(methods)
        print("\n7) CONSENSUS RANKING")
        print("Agent   AvgRank   (Win%,Elo,Tourn,BT)")
        for A in sorted(avg_rank, key=lambda x: avg_rank[x]):
            ranks = ",".join(str(int(r)) for r in all_ranks[A])
            print(f"  {A:<7} {avg_rank[A]:7.2f}    [{ranks}]")

        champ = min(avg_rank, key=lambda x: avg_rank[x])
        print(f"\n🏆  STRONGEST AGENT: {champ} (avg rank {avg_rank[champ]:.2f})\n")


def main():
    try:
        # if you have 4-player matches, set group_size=4
        analyzer = AgentRankingAnalyzer('results.csv', group_size=3)
        analyzer.print_comprehensive_analysis()
    except FileNotFoundError:
        print("Error: results.csv not found.")

if __name__ == "__main__":
    main()
