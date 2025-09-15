# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def load_plugin_statistics(stats_file: str) -> Dict[str, Any]:
    """Load plugin statistics from JSON file."""
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        LOG.warning(f"Failed to load {stats_file}: {e}")
        return {}


def aggregate_sparse_ratios(data_dir: str, output_file: str = None) -> Dict[str, Any]:
    """
    Aggregate sparse attention statistics from all plugin_statistics.json files in directory.
    
    Args:
        data_dir: Directory containing *_plugin_statistics.json files
        output_file: Optional path to save aggregated results (defaults to data_dir/sparsity_summary.json)
    
    Returns:
        Dictionary containing aggregated statistics
    """
    # Find plugin statistics files
    stats_files = glob.glob(os.path.join(data_dir, "*_plugin_statistics.json"))
    
    if not stats_files:
        LOG.info("No plugin statistics files found - skipping sparse ratio aggregation")
        return None
    
    LOG.info(f"Found {len(stats_files)} plugin statistics files")
    
    # Load plugin statistics
    plugin_stats = {}
    for stats_file in stats_files:
        file_name = os.path.basename(stats_file).replace("_plugin_statistics.json", "")
        stats = load_plugin_statistics(stats_file)
        if stats:
            plugin_stats[file_name] = stats
            LOG.info(f"  Loaded stats for: {file_name}")
    
    if not plugin_stats:
        return None
    
    # Collect method-specific sparse ratio data with phase separation
    method_sparse_ratios = {}  # {method_name: {phase: [ratios...]}}
    attention_methods = set()
    
    for file_name, stats in plugin_stats.items():
        method = stats.get("method", "unknown")
        attention_methods.add(method)
        
        # Initialize method structure if not exists
        if method not in method_sparse_ratios:
            method_sparse_ratios[method] = {}
        
        # Overall sparse ratio
        if "overall_average_sparse_ratio" in stats and stats["overall_average_sparse_ratio"] is not None:
            if "overall" not in method_sparse_ratios[method]:
                method_sparse_ratios[method]["overall"] = []
            method_sparse_ratios[method]["overall"].append(stats["overall_average_sparse_ratio"])
        
        # Phase-specific ratios
        if "prefill_average_sparse_ratio" in stats and stats["prefill_average_sparse_ratio"] is not None:
            if "prefill" not in method_sparse_ratios[method]:
                method_sparse_ratios[method]["prefill"] = []
            method_sparse_ratios[method]["prefill"].append(stats["prefill_average_sparse_ratio"])
        
        if "decode_average_sparse_ratio" in stats and stats["decode_average_sparse_ratio"] is not None:
            if "decode" not in method_sparse_ratios[method]:
                method_sparse_ratios[method]["decode"] = []
            method_sparse_ratios[method]["decode"].append(stats["decode_average_sparse_ratio"])
    
    # Build structured summary
    method_data = {}
    for method_name, phase_data in method_sparse_ratios.items():
        method_info = {
            "phases": list(phase_data.keys()),
            "phase_sparse_ratios": {},
            "phase_statistics": {}
        }
        
        for phase_name, ratios in phase_data.items():
            avg_ratio = sum(ratios) / len(ratios)
            min_ratio = min(ratios)
            max_ratio = max(ratios)
            
            method_info["phase_sparse_ratios"][phase_name] = avg_ratio
            method_info["phase_statistics"][phase_name] = {
                "average": avg_ratio,
                "min": min_ratio,
                "max": max_ratio,
                "count": len(ratios)
            }
        
        method_data[method_name] = method_info
    
    # Create summary
    summary = {
        "evaluation_summary": {
            "results_directory": data_dir,
            "timestamp": datetime.datetime.now().isoformat(),
            "num_files_processed": len(plugin_stats)
        },
        "sparse_attention_summary": {
            "total_files_with_stats": len(plugin_stats),
            "attention_methods": list(attention_methods),
            "method_phase_data": method_data
        }
    }
    
    # Save to file
    if output_file is None:
        output_file = os.path.join(data_dir, "sparsity_summary.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    LOG.info(f"\n✓ Aggregated sparsity summary saved to: {output_file}")
    
    # Print summary
    for method_name, method_info in method_data.items():
        LOG.info(f"\n{method_name}:")
        for phase_name, stats in method_info["phase_statistics"].items():
            LOG.info(f"  {phase_name}: {stats['average']:.4f} (min: {stats['min']:.4f}, max: {stats['max']:.4f}, n={stats['count']})")
    
    return summary


def main():
    """Command-line interface for aggregating sparsity statistics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate sparsity statistics from plugin files")
    parser.add_argument("data_dir", help="Directory containing *_plugin_statistics.json files")
    parser.add_argument("--output", help="Output file path (default: data_dir/sparsity_summary.json)")
    
    args = parser.parse_args()
    
    aggregate_sparse_ratios(args.data_dir, args.output)


if __name__ == "__main__":
    main()

