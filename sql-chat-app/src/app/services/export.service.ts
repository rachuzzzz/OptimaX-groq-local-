import { Injectable } from '@angular/core';
import { saveAs } from 'file-saver';

@Injectable({
  providedIn: 'root'
})
export class ExportService {

  constructor() { }

  /**
   * Export data to CSV format
   */
  exportToCSV(data: any[], filename: string = 'query_results.csv'): void {
    if (!data || data.length === 0) {
      console.warn('No data to export');
      return;
    }

    try {
      const headers = Object.keys(data[0]);
      const csvRows = [];

      // Add headers
      csvRows.push(headers.map(h => this.escapeCSVValue(h)).join(','));

      // Add data rows
      for (const row of data) {
        const values = headers.map(header => {
          const value = row[header];
          return this.escapeCSVValue(value);
        });
        csvRows.push(values.join(','));
      }

      const csvContent = csvRows.join('\n');
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      saveAs(blob, filename);
    } catch (error) {
      console.error('Error exporting to CSV:', error);
      throw error;
    }
  }

  /**
   * Export data to JSON format
   */
  exportToJSON(data: any[], filename: string = 'query_results.json'): void {
    if (!data || data.length === 0) {
      console.warn('No data to export');
      return;
    }

    try {
      const jsonContent = JSON.stringify(data, null, 2);
      const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
      saveAs(blob, filename);
    } catch (error) {
      console.error('Error exporting to JSON:', error);
      throw error;
    }
  }

  /**
   * Export SQL query to .sql file
   */
  exportSQLQuery(sqlQuery: string, filename: string = 'query.sql'): void {
    if (!sqlQuery) {
      console.warn('No SQL query to export');
      return;
    }

    try {
      const blob = new Blob([sqlQuery], { type: 'text/plain;charset=utf-8;' });
      saveAs(blob, filename);
    } catch (error) {
      console.error('Error exporting SQL:', error);
      throw error;
    }
  }

  /**
   * Escape special characters for CSV
   */
  private escapeCSVValue(value: any): string {
    if (value === null || value === undefined) {
      return '';
    }

    const stringValue = String(value);

    // If value contains comma, quote, or newline, wrap in quotes and escape quotes
    if (stringValue.includes(',') || stringValue.includes('"') || stringValue.includes('\n')) {
      return `"${stringValue.replace(/"/g, '""')}"`;
    }

    return stringValue;
  }

  /**
   * Copy text to clipboard
   */
  copyToClipboard(text: string): Promise<void> {
    return navigator.clipboard.writeText(text);
  }

  /**
   * Format data as markdown table for clipboard
   */
  formatAsMarkdownTable(data: any[]): string {
    if (!data || data.length === 0) {
      return '';
    }

    const headers = Object.keys(data[0]);
    const rows = [];

    // Add header row
    rows.push('| ' + headers.join(' | ') + ' |');
    rows.push('| ' + headers.map(() => '---').join(' | ') + ' |');

    // Add data rows
    for (const row of data) {
      const values = headers.map(header => String(row[header] ?? ''));
      rows.push('| ' + values.join(' | ') + ' |');
    }

    return rows.join('\n');
  }
}
