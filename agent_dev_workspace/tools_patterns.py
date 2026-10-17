import os
import tempfile
import pandas as pd
from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import PAMI.frequentPattern.basic.FPGrowth as FPGrowth
import PAMI.frequentPattern.topk.FAE as FAE
import PAMI.frequentPattern.maximal.MaxFPGrowth as MaxFPGrowth
from PAMI.extras.convert import DF2DB


def make_tools(session):
    """Factory function to instantiate tools with access to the shared session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "tfidf",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Mines frequent patterns for a specific category's documents.

        Filters category-specific vocabulary using variance, tfidf, or term_frequency,
        converts the surviving terms into a transactional database, and mines patterns
        using FPGrowth, Top-K (FAE), or MaxFPGrowth.

        Args:
            category_name: Category to filter session.dataframe by.
            filtering_method: Vocabulary filtering approach ('variance', 'tfidf', or 'term_frequency').
            algorithm: Pattern mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-k value (required for 'topk').

        Returns:
            Dict containing pattern mining summary and status.
        """
        # Validate parameters at the very beginning
        valid_filtering_methods = ["variance", "tfidf", "term_frequency"]
        if filtering_method not in valid_filtering_methods:
            return {"error": f"Invalid filtering_method: '{filtering_method}'. Must be one of {valid_filtering_methods}"}

        valid_algorithms = ["fpgrowth", "topk", "maxfpgrowth"]
        if algorithm not in valid_algorithms:
            return {"error": f"Invalid algorithm: '{algorithm}'. Must be one of {valid_algorithms}"}

        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            return {"error": f"min_sup is required for algorithm '{algorithm}'."}

        if algorithm == "topk" and k is None:
            return {"error": "k is required for algorithm 'topk'."}

        args_recorded = {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k,
        }

        # Check dataframe and column existence
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session."}

        if "category_name" not in session.dataframe.columns:
            return {"error": "Column 'category_name' missing from session.dataframe."}

        if "text" not in session.dataframe.columns:
            return {"error": "Column 'text' missing from session.dataframe."}

        # Filter session.dataframe to category_name
        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        if df_cat.empty:
            return {"error": f"No data found for category: '{category_name}'"}

        # Step 1: Category-local DTM with stop_words="english"
        texts = df_cat["text"].fillna("").tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts)
        except ValueError:
            dtm = None

        feature_names = vectorizer.get_feature_names_out() if dtm is not None else []

        if dtm is None or dtm.shape[1] == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", args_recorded, summary)
            return summary

        # Step 2: Vocabulary filtering using inclusive operators >= lower and <= upper
        dtm_dense = dtm.toarray()
        dtm_df = pd.DataFrame(dtm_dense, columns=feature_names)

        if filtering_method == "variance":
            scores = dtm_df.var(axis=0)
            low = scores.quantile(0.05)
            high = scores.quantile(0.95)
            kept_cols = scores[(scores >= low) & (scores <= high)].index.tolist()
        elif filtering_method == "term_frequency":
            scores = dtm_df.sum(axis=0)
            low = scores.quantile(0.05)
            high = scores.quantile(0.95)
            kept_cols = scores[(scores >= low) & (scores <= high)].index.tolist()
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(dtm)
            scores = pd.Series(tfidf.toarray().mean(axis=0), index=feature_names)
            cutoff = scores.quantile(0.20)
            kept_cols = scores[scores >= cutoff].index.tolist()

        # Step 3: Check zero surviving terms
        if not kept_cols:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "pattern_count": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", args_recorded, summary)
            return summary

        filtered_df = dtm_df[kept_cols]

        # Step 4: Transactional DB conversion and PAMI mining
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "transactional.db")

            # DF2DB constructor takes DataFrame, convert2TransactionalDatabase takes oFile, condition='>=', thresholdValue=1
            converter = DF2DB.DF2DB(filtered_df)
            converter.convert2TransactionalDatabase(db_file, ">=", 1)

            # Mine patterns based on algorithm
            if algorithm == "fpgrowth":
                miner = FPGrowth.FPGrowth(db_file, minSup=min_sup)
                miner.startMine()
                patterns_df = miner.getPatternsAsDataFrame()
            elif algorithm == "topk":
                miner = FAE.FAE(db_file, k=k)
                miner.startMine()
                patterns_df = miner.getPatternsAsDataFrame()
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup)
                miner.startMine()
                patterns_df = miner.getPatternsAsDataFrame()

        # Step 5: Format outputs, sort by support descending, preview top 20, and set full_report
        patterns_list = []

        if patterns_df is not None and not patterns_df.empty:
            cols = patterns_df.columns.tolist()
            pattern_col = cols[0]
            sup_col = cols[1] if len(cols) > 1 else cols[0]

            for _, row in patterns_df.iterrows():
                p_val = str(row[pattern_col])
                s_val = int(row[sup_col]) if sup_col != pattern_col else 0
                patterns_list.append({"pattern": p_val, "support": s_val})

            # Sort patterns by support from highest to lowest
            patterns_list.sort(key=lambda x: x["support"], reverse=True)

        full_patterns_df = pd.DataFrame(patterns_list) if patterns_list else pd.DataFrame(columns=["pattern", "support"])
        pattern_count = len(patterns_list)
        preview_patterns = patterns_list[:20]  # Top 20 preview

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "pattern_count": pattern_count,
            "patterns": preview_patterns,
        }

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list
        session.store_result("mine_patterns_tool", args_recorded, summary, full_report=full_patterns_df)

        return summary

    return [mine_patterns_tool]
