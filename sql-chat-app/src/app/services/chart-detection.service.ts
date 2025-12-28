import { Injectable } from '@angular/core';
import { ChartType } from 'chart.js';
import { ChartData } from '../components/chart-visualization/chart-visualization';

@Injectable({
  providedIn: 'root'
})
export class ChartDetectionService {
  /**
   * OptimaX Visualization Workflow - Deterministic Chart Rendering Service
   * ========================================================================
   *
   * This service is responsible ONLY for deterministic chart rendering.
   * Per the OptimaX specification:
   *
   * - LLM classifies intent and suggests chart types via metadata tags
   * - User selects chart type from suggestions
   * - This service renders the chart using predefined templates
   *
   * NO heuristic detection or AI decision-making happens here.
   * Role: Parse data → Render chart deterministically
   */

  constructor() { }

  /**
   * Parses SQL results into chart data format
   */
  parseToChartData(results: any[], chartType?: ChartType): ChartData {
    if (!results || results.length === 0) {
      return {
        type: 'bar',
        labels: [],
        datasets: []
      };
    }

    const keys = Object.keys(results[0]);
    const labelKey = keys[0]; // First column as labels
    const dataKeys = keys.slice(1); // Remaining columns as data

    const labels = results.map(row => String(row[labelKey]));

    // Create datasets for each data column
    const datasets = dataKeys.map((key, index) => {
      const data = results.map(row => Number(row[key]) || 0);

      return {
        label: this.formatLabel(key),
        data: data,
        backgroundColor: this.getColors(results.length, chartType),
        borderColor: 'rgba(255, 215, 0, 1)',
        borderWidth: 2
      };
    });

    return {
      type: chartType || 'bar',
      labels,
      datasets
    };
  }

  /**
   * Generates gold-themed colors for charts
   */
  private getColors(count: number, chartType?: ChartType): string[] {
    const goldColors = [
      'rgba(255, 215, 0, 0.8)',     // Gold
      'rgba(212, 175, 55, 0.8)',    // Dark Gold
      'rgba(255, 223, 0, 0.8)',     // Light Gold
      'rgba(184, 134, 11, 0.8)',    // Dark Goldenrod
      'rgba(255, 185, 15, 0.8)',    // Amber
      'rgba(218, 165, 32, 0.8)',    // Goldenrod
      'rgba(238, 232, 170, 0.7)',   // Pale Goldenrod
      'rgba(189, 183, 107, 0.8)',   // Dark Khaki
    ];

    // For pie/doughnut charts, return array of different colors
    if (chartType === 'pie' || chartType === 'doughnut') {
      return Array.from({ length: count }, (_, i) => goldColors[i % goldColors.length]);
    }

    // For bar/line charts, use gold gradient
    return goldColors.slice(0, Math.min(count, goldColors.length));
  }

  /**
   * Formats column names for better display
   */
  private formatLabel(label: string): string {
    // Convert snake_case or camelCase to Title Case
    return label
      .replace(/_/g, ' ')
      .replace(/([A-Z])/g, ' $1')
      .trim()
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  }

}
