import { Injectable } from '@angular/core';

export interface QueryExample {
  title: string;
  query: string;
  category: 'basic' | 'aggregation' | 'joins' | 'analytics' | 'time';
  description: string;
}

export interface DatabaseExamples {
  databaseName: string;
  examples: QueryExample[];
}

@Injectable({
  providedIn: 'root'
})
export class QueryExamplesService {

  constructor() { }

  /**
   * Get query examples for postgres_air database
   */
  getPostgresAirExamples(): QueryExample[] {
    return [
      // Basic queries
      {
        title: 'List All Airports',
        query: 'Show me all airports',
        category: 'basic',
        description: 'View all airports in the database'
      },
      {
        title: 'Count Total Flights',
        query: 'How many flights are there?',
        category: 'basic',
        description: 'Get total number of flights'
      },
      {
        title: 'Recent Bookings',
        query: 'Show me the 10 most recent bookings',
        category: 'basic',
        description: 'View latest booking records'
      },

      // Aggregation queries
      {
        title: 'Top Routes by Flight Count',
        query: 'What are the top 10 routes by number of flights?',
        category: 'aggregation',
        description: 'Most popular flight routes'
      },
      {
        title: 'Average Booking Price',
        query: 'What is the average booking price?',
        category: 'aggregation',
        description: 'Calculate mean booking cost'
      },
      {
        title: 'Flights by Status',
        query: 'Count flights by status',
        category: 'aggregation',
        description: 'Distribution of flight statuses'
      },

      // Join queries
      {
        title: 'Passengers with Bookings',
        query: 'Show passengers with their booking information',
        category: 'joins',
        description: 'Multi-table passenger data'
      },
      {
        title: 'Flight Details with Aircraft',
        query: 'List flights with aircraft information',
        category: 'joins',
        description: 'Flights joined with aircraft models'
      },
      {
        title: 'Bookings with Passenger Names',
        query: 'Show bookings with passenger names',
        category: 'joins',
        description: 'Complete booking and passenger details'
      },

      // Analytics queries
      {
        title: 'Busiest Airports',
        query: 'Which airports have the most departures?',
        category: 'analytics',
        description: 'Airports ranked by activity'
      },
      {
        title: 'Frequent Flyer Statistics',
        query: 'Show frequent flyer distribution by airline',
        category: 'analytics',
        description: 'Loyalty program analysis'
      },

      // Time-based queries
      {
        title: 'Flights Today',
        query: 'Show me flights scheduled for today',
        category: 'time',
        description: 'Current day flight schedule'
      },
      {
        title: 'Delayed Flights',
        query: 'Show delayed flights',
        category: 'time',
        description: 'Flights with delayed status'
      }
    ];
  }

  /**
   * Get query examples for traffic_db (US Accidents) database
   */
  getTrafficAccidentsExamples(): QueryExample[] {
    return [
      // Basic queries
      {
        title: 'Total Accidents',
        query: 'How many accidents are in the database?',
        category: 'basic',
        description: 'Total accident count'
      },
      {
        title: 'Recent Accidents',
        query: 'Show me the 10 most recent accidents',
        category: 'basic',
        description: 'Latest accident records'
      },
      {
        title: 'Accidents by State',
        query: 'Count accidents by state',
        category: 'basic',
        description: 'State-level accident distribution'
      },

      // Aggregation queries
      {
        title: 'Top 10 States',
        query: 'Show me the top 10 states with most accidents',
        category: 'aggregation',
        description: 'States ranked by accident count'
      },
      {
        title: 'Accidents by Severity',
        query: 'Count accidents by severity level',
        category: 'aggregation',
        description: 'Distribution of accident severity'
      },
      {
        title: 'Top Cities',
        query: 'What are the top 20 cities with most accidents?',
        category: 'aggregation',
        description: 'Most accident-prone cities'
      },

      // Analytics queries
      {
        title: 'Weather Impact',
        query: 'Show accidents by weather condition',
        category: 'analytics',
        description: 'Weather-related accident analysis'
      },
      {
        title: 'Severity by State',
        query: 'What is the average severity by state?',
        category: 'analytics',
        description: 'State accident severity analysis'
      },
      {
        title: 'Road Features Analysis',
        query: 'Count accidents at traffic signals vs junctions',
        category: 'analytics',
        description: 'Road infrastructure impact'
      },

      // Time-based queries
      {
        title: 'Accidents in 2022',
        query: 'Show accidents from 2022',
        category: 'time',
        description: 'Year-specific accident data'
      },
      {
        title: 'Monthly Trends',
        query: 'Show accidents by month in 2021',
        category: 'time',
        description: 'Monthly accident patterns'
      },
      {
        title: 'Peak Hours',
        query: 'What hours have the most accidents?',
        category: 'time',
        description: 'Time-of-day analysis'
      },

      // Specific conditions
      {
        title: 'Rain Accidents',
        query: 'Show accidents during rain',
        category: 'analytics',
        description: 'Rain-related accidents'
      },
      {
        title: 'California Severe',
        query: 'Show severe accidents in California',
        category: 'analytics',
        description: 'State-specific severe accidents'
      }
    ];
  }

  /**
   * Get examples by category
   */
  getExamplesByCategory(examples: QueryExample[], category: string): QueryExample[] {
    return examples.filter(ex => ex.category === category);
  }

  /**
   * Auto-detect database and return appropriate examples
   */
  getExamplesForDatabase(databaseName: string): QueryExample[] {
    const normalizedName = databaseName.toLowerCase();

    if (normalizedName.includes('postgres_air') || normalizedName.includes('flight') || normalizedName.includes('air')) {
      return this.getPostgresAirExamples();
    } else if (normalizedName.includes('traffic') || normalizedName.includes('accident')) {
      return this.getTrafficAccidentsExamples();
    }

    // Default to postgres_air examples
    return this.getPostgresAirExamples();
  }
}
