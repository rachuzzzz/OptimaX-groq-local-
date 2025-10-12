import { Injectable } from '@angular/core';
import { ChartType } from 'chart.js';
import { ChartData } from '../components/chart-visualization/chart-visualization';

@Injectable({
  providedIn: 'root'
})
export class ChartDetectionService {

  constructor() { }

  /**
   * Determines if a SQL query result should be visualized
   */
  shouldVisualize(query: string, results: any[]): boolean {
    if (!results || results.length === 0) {
      return false;
    }

    const queryUpper = query.toUpperCase();
    const hasAggregation = /COUNT|SUM|AVG|MAX|MIN/i.test(query);
    const hasGroupBy = /GROUP\s+BY/i.test(query);
    const firstRow = results[0];
    const columnCount = Object.keys(firstRow).length;

    // Visualize if:
    // 1. Has aggregation functions OR
    // 2. Has GROUP BY clause OR
    // 3. Has exactly 2 columns (label + value pattern)
    return (hasAggregation || hasGroupBy) && columnCount >= 2 && columnCount <= 4;
  }

  /**
   * Detects the appropriate chart type based on query
   */
  detectChartType(query: string, results: any[]): ChartType {
    const queryUpper = query.toUpperCase();

    // Time series detection (line chart)
    if (/DATE|TIME|MONTH|YEAR|DAY/i.test(query) && /GROUP\s+BY/i.test(query)) {
      return 'line';
    }

    // Distribution/Proportion (pie/doughnut chart)
    if (/LIMIT\s+\d+/i.test(query) && Object.keys(results[0] || {}).length === 2) {
      const limitMatch = query.match(/LIMIT\s+(\d+)/i);
      const limit = limitMatch ? parseInt(limitMatch[1]) : 0;
      if (limit <= 8) {
        return 'doughnut';
      }
    }

    // COUNT or SUM aggregations (bar chart)
    if (/COUNT|SUM/i.test(query)) {
      return 'bar';
    }

    // AVG (line chart for trends)
    if (/AVG/i.test(query)) {
      return 'line';
    }

    // Default to bar chart
    return 'bar';
  }

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

  /**
   * Analyzes query complexity for visualization recommendations
   */
  getVisualizationRecommendation(query: string, results: any[]): {
    shouldVisualize: boolean;
    chartType: ChartType;
    confidence: number;
    reason: string;
  } {
    const shouldViz = this.shouldVisualize(query, results);
    const chartType = this.detectChartType(query, results);

    let confidence = 0;
    let reason = '';

    if (!shouldViz) {
      reason = 'Query results are not suitable for visualization';
      return { shouldVisualize: false, chartType: 'bar', confidence: 0, reason };
    }

    // Calculate confidence based on query patterns
    const queryUpper = query.toUpperCase();

    if (/GROUP\s+BY/i.test(query) && /COUNT|SUM/i.test(query)) {
      confidence = 0.9;
      reason = 'Aggregated data with grouping - excellent for charts';
    } else if (/DATE|TIME/i.test(query) && /GROUP\s+BY/i.test(query)) {
      confidence = 0.95;
      reason = 'Time series data - perfect for line charts';
    } else if (Object.keys(results[0]).length === 2) {
      confidence = 0.8;
      reason = 'Two-column data pattern - good for visualization';
    } else {
      confidence = 0.6;
      reason = 'Data can be visualized with moderate effectiveness';
    }

    return { shouldVisualize: true, chartType, confidence, reason };
  }
}
