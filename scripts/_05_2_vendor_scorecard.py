import os
from datetime import datetime
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

class VendorAnalyticsEngine:
    """
    An analytics engine that computes engagement and business metrics for a vendor channel,
    combining post metadata (Dates, Views) with NER-extracted price and product data.

    Attributes:
        posts (list): A list of dictionaries, each representing a Telegram post with at least:
                        - 'Date' (ISO format string)
                        - 'Views' (int)
                        - 'Message' (str)
                        - 'NER_Prices' (list of float)
                        - 'NER_Products' (list of str)
    """

    def __init__(self, posts):
        """
        Initialises the VendorAnalyticsEngine with a list of posts.
        Args:
            posts (list): List of post dicts with metadata and NER output.
        """

        self.posts = posts

    @staticmethod
    def clean_list_column(column):
        """
        Cleans a pandas Series by splitting comma-separated strings into stripped lists.
        Useful for turning 'value1, value2' into ['value1', 'value2'].

        Args:
            column (pd.Series): A pandas Series of stringified list values.        
        
        Returns:
            pd.Series: A Series with lists of cleaned items.
        """

        return column.apply(lambda x: [i.strip() for i in str(x).split(",") if i.strip()])
    
    @staticmethod

    def generate_leaderboard(vendor_engines, vendor_names=None, sort_by="Lending Score", visualise=False, top_n=10):
        """
        Generates a leaderboard DataFrame from multiple VendorAnalyticsEngine instances.

        Args:
            vendor_engines (list): List of VendorAnalyticsEngine objects.
            vendor_names (list, optional): Names associated with each vendor engine.
            sort_by (str): Metric to sort the leaderboard by.
            visualise (bool): If True, renders a bar chart of top N vendors.
            top_n (int): Number of top vendors to show in chart.

        Returns:
            pd.DataFrame: Leaderboard with vendor metrics.
        """

        if vendor_names and len(vendor_names) != len(vendor_engines):
            raise ValueError("Length of vendor_names must match vendor_engines")

        records = []
        for i, engine in enumerate(vendor_engines):
            name = vendor_names[i] if vendor_names else f"Vendor_{i+1}"
            top_post = engine.top_performing_post()
            records.append({
                "Vendor": name,
                "Posts/Week": engine.posting_frequency(),
                "Avg Views/Post": engine.average_views(),
                "Avg Price (ETB)": engine.average_price(),
                "Top Product": top_post.get("product"),
                "Top Price": top_post.get("price"),
                "Top Views": top_post.get("Views"),
                "Lending Score": engine.compute_lending_score()
            })

        leaderboard = pd.DataFrame(records).sort_values(sort_by, ascending=False).reset_index(drop=True)

        df_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'vendor_scorecard.csv')
        leaderboard.to_csv(df_path, index=False)

        rel_df_path = os.path.relpath(df_path, os.getcwd())
        print(f"\nLeaderboard successfully saved to: {rel_df_path}")

        if visualise:

            bars = leaderboard.head(top_n)
            colors = sns.color_palette("hls", len(bars))
            
            plt.figure(figsize=(12, 6))
            plt.barh(bars["Vendor"], bars[sort_by], color=colors)
            plt.gca().invert_yaxis()
            plt.xlabel(sort_by)
            plt.title(f"Top {top_n} Vendors by {sort_by}")
            plt.tight_layout()

            # Save plot to scorecard plot directory
            plot_dir = os.path.join(os.path.dirname(os.getcwd()), 'scorecard plot')
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, "vendor_leaderboard.png")
            plt.savefig(plot_path)

            rel_plot_path = os.path.relpath(plot_dir, os.getcwd())
            print(f"\nPlot saved to: {rel_plot_path}")

            plt.show()

        return leaderboard


    def posting_frequency(self):
        """
        Calculates average posting frequency (posts per week).

        Returns:
            float: Posts per week.
        """
        weeks = defaultdict(int)
        for post in self.posts:
            try:
                dt = datetime.fromisoformat(post["Date"])
            except (ValueError, TypeError):
                continue
            key = dt.strftime("%Y-W%U")
            weeks[key] += 1
        return sum(weeks.values()) / len(weeks) if weeks else 0

    def average_views(self):
        """
        Computes the average number of views per post.

        Returns:
            float: Average views.
        """
        return sum(p["Views"] for p in self.posts) / len(self.posts) if self.posts else 0

    def top_performing_post(self):
        """
        Finds the post with the highest view count.

        Returns:
            dict: A dictionary with 'Views', 'Message', 'product', 'price' fields.
        """
        top_post = max(self.posts, key=lambda p: p["Views"], default=None)
        if not top_post:
            return {}
        return {
            "Views": top_post["Views"],
            "Message": top_post["Message"],
            "product": top_post.get("NER_Products", ["N/A"])[0] if top_post.get("NER_Products") else "N/A",
            "price": top_post.get("NER_Prices", ["N/A"])[0] if top_post.get("NER_Prices") else "N/A"
        }

    def average_price(self):
        """
        Computes the average listed price across all posts.

        Returns:
            float: Average price or 0 if no prices found.
        """
        def extract_numeric(p):
            try:
                num = "".join(c for c in str(p) if c.isdigit() or c == ".")
                return float(num) if num.count(".") <= 1 else None
            except:
                return None
        all_prices = [extract_numeric(p) for post in self.posts for p in post.get("NER_Prices", [])]
        clean = [p for p in all_prices if p]
        return sum(clean) / len(clean) if clean else 0


    def compute_lending_score(self, weights=None, return_components=False):
        """
        Computes a final lending score based on customisable weights.

        Args:
            weights (dict, optional): Dictionary with 'Views', 'frequency', and 'price' keys.
                                        Default: {'Views': 0.5, 'frequency': 0.3, 'price': 0.2}

        Returns:
            float: Lending score.
        """
        weights = weights or {"Views": 0.5, "frequency": 0.3, "price": 0.2}

        view_score = self.average_views() * weights["Views"]
        freq_score = self.posting_frequency() * weights["frequency"]
        price_score = self.average_price() * weights["price"]
        score = view_score + freq_score + price_score

        if return_components:
            return {
                "Views": view_score,
                "posts": freq_score,
                "price": price_score,
                "total": score
            }

        return score


    def summary(self):
        """
        Generates a textual summary of key vendor metrics.

        Returns:
            str: Summary report.
        """
        top_post = self.top_performing_post()
        return f"""Vendor Summary:
        - Posts/week: {self.posting_frequency():.2f}
        - Avg Views/Post: {self.average_views():,.0f}
        - Avg Price: {self.average_price():,.0f} birr
        - Top Post: “{top_post.get('product')}” for {top_post.get('price')} birr ({top_post.get('Views')} Views)
        - Lending Score: {self.compute_lending_score():,.2f}"""
