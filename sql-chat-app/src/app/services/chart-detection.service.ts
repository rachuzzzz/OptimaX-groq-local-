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
        borderColor: this.getBorderColors(results.length, chartType),
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
   * Generates vibrant, diverse colors for charts
   */
  private getColors(count: number, chartType?: ChartType): string[] {
    // Vibrant color palette with excellent contrast and visual appeal
    const vibrantColors = [
      'rgba(59, 130, 246, 0.8)',    // Bright Blue
      'rgba(16, 185, 129, 0.8)',    // Emerald Green
      'rgba(249, 115, 22, 0.8)',    // Vibrant Orange
      'rgba(139, 92, 246, 0.8)',    // Purple
      'rgba(236, 72, 153, 0.8)',    // Pink
      'rgba(245, 158, 11, 0.8)',    // Amber
      'rgba(20, 184, 166, 0.8)',    // Teal
      'rgba(239, 68, 68, 0.8)',     // Red
      'rgba(168, 85, 247, 0.8)',    // Violet
      'rgba(34, 197, 94, 0.8)',     // Green
      'rgba(251, 191, 36, 0.8)',    // Yellow
      'rgba(14, 165, 233, 0.8)',    // Sky Blue
      'rgba(217, 70, 239, 0.8)',    // Fuchsia
      'rgba(99, 102, 241, 0.8)',    // Indigo
      'rgba(234, 179, 8, 0.8)',     // Gold
      'rgba(244, 63, 94, 0.8)',     // Rose
    ];

    // For pie/doughnut charts, return array of different colors for each slice
    if (chartType === 'pie' || chartType === 'doughnut') {
      return Array.from({ length: count }, (_, i) => vibrantColors[i % vibrantColors.length]);
    }

    // For bar/line charts with multiple datasets, assign different colors
    return vibrantColors.slice(0, Math.min(count, vibrantColors.length));
  }

  /**
   * Generates border colors (slightly more opaque than backgrounds)
   */
  private getBorderColors(count: number, chartType?: ChartType): string[] {
    const borderColors = [
      'rgba(59, 130, 246, 1)',      // Bright Blue
      'rgba(16, 185, 129, 1)',      // Emerald Green
      'rgba(249, 115, 22, 1)',      // Vibrant Orange
      'rgba(139, 92, 246, 1)',      // Purple
      'rgba(236, 72, 153, 1)',      // Pink
      'rgba(245, 158, 11, 1)',      // Amber
      'rgba(20, 184, 166, 1)',      // Teal
      'rgba(239, 68, 68, 1)',       // Red
      'rgba(168, 85, 247, 1)',      // Violet
      'rgba(34, 197, 94, 1)',       // Green
      'rgba(251, 191, 36, 1)',      // Yellow
      'rgba(14, 165, 233, 1)',      // Sky Blue
      'rgba(217, 70, 239, 1)',      // Fuchsia
      'rgba(99, 102, 241, 1)',      // Indigo
      'rgba(234, 179, 8, 1)',       // Gold
      'rgba(244, 63, 94, 1)',       // Rose
    ];

    if (chartType === 'pie' || chartType === 'doughnut') {
      return Array.from({ length: count }, (_, i) => borderColors[i % borderColors.length]);
    }

    return borderColors.slice(0, Math.min(count, borderColors.length));
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
