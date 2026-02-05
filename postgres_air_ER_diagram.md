# postgres_air Database - ER Diagram

## Mermaid ER Diagram

```mermaid
erDiagram
    %% ==========================================
    %% POSTGRES_AIR DATABASE - ER DIAGRAM
    %% ==========================================

    %% CORE ENTITIES (Master Data)
    AIRCRAFT {
        text code PK "Aircraft type code"
        text model "Aircraft model name"
        numeric range "Maximum flight range"
        integer class "Aircraft class"
        numeric velocity "Cruise velocity"
    }

    AIRPORT {
        char airport_code PK "3-letter IATA code"
        text airport_name "Full airport name"
        text city "City location"
        text airport_tz "Timezone"
        text continent "Continent"
        text iso_country "ISO country code"
        text iso_region "ISO region code"
        boolean intnl "International flag"
        timestamp update_ts "Last update"
    }

    FREQUENT_FLYER {
        integer frequent_flyer_id PK
        text first_name
        text last_name
        text title
        text card_num "Loyalty card number"
        integer level "Membership tier"
        integer award_points "Accumulated points"
        text email
        text phone
        timestamp update_ts
    }

    %% USER/ACCOUNT ENTITIES
    ACCOUNT {
        integer account_id PK
        text login "Username"
        text first_name
        text last_name
        integer frequent_flyer_id FK
        timestamp update_ts
    }

    PHONE {
        integer phone_id PK
        integer account_id FK
        text phone "Phone number"
        text phone_type "Mobile/Home/Work"
        boolean primary_phone "Is primary"
        timestamp update_ts
    }

    %% BOOKING ENTITIES
    BOOKING {
        bigint booking_id PK
        text booking_ref "Reference code"
        text booking_name "Booking holder name"
        integer account_id FK
        text email
        text phone
        numeric price "Total price"
        timestamp update_ts
    }

    PASSENGER {
        integer passenger_id PK
        integer booking_id FK
        text booking_ref
        integer passenger_no "Passenger sequence"
        text first_name
        text last_name
        integer account_id FK
        integer age
        timestamp update_ts
    }

    %% FLIGHT OPERATIONS
    FLIGHT {
        integer flight_id PK
        text flight_no "Flight number"
        timestamp scheduled_departure
        timestamp scheduled_arrival
        char departure_airport FK "3-letter code"
        char arrival_airport FK "3-letter code"
        text status "Scheduled/Departed/Arrived"
        char aircraft_code FK
        timestamp actual_departure
        timestamp actual_arrival
        timestamp update_ts
    }

    BOOKING_LEG {
        integer booking_leg_id PK
        integer booking_id FK
        integer flight_id FK
        integer leg_num "Segment number"
        boolean is_returning "Return trip flag"
        timestamp update_ts
    }

    BOARDING_PASS {
        integer pass_id PK
        bigint passenger_id FK
        bigint booking_leg_id FK
        text seat "Seat assignment"
        timestamp boarding_time
        boolean precheck "TSA PreCheck"
        timestamp update_ts
    }

    %% ==========================================
    %% RELATIONSHIPS
    %% ==========================================

    %% Account relationships
    FREQUENT_FLYER ||--o{ ACCOUNT : "has"
    ACCOUNT ||--o{ PHONE : "has"
    ACCOUNT ||--o{ BOOKING : "makes"
    ACCOUNT ||--o{ PASSENGER : "travels_as"

    %% Booking relationships
    BOOKING ||--|{ PASSENGER : "includes"
    BOOKING ||--|{ BOOKING_LEG : "has_legs"

    %% Flight relationships
    BOOKING_LEG }|--|| FLIGHT : "on_flight"
    FLIGHT }|--|| AIRCRAFT : "uses"
    FLIGHT }|--|| AIRPORT : "departs_from"
    FLIGHT }|--|| AIRPORT : "arrives_at"

    %% Boarding relationships
    BOARDING_PASS }|--|| PASSENGER : "for_passenger"
    BOARDING_PASS }|--|| BOOKING_LEG : "for_leg"
```

## Relationships Summary

| From Table | Relationship | To Table | FK Column | Description |
|------------|--------------|----------|-----------|-------------|
| `account` | N:1 | `frequent_flyer` | `frequent_flyer_id` | Account linked to loyalty program |
| `phone` | N:1 | `account` | `account_id` | Multiple phones per account |
| `booking` | N:1 | `account` | `account_id` | Bookings made by account |
| `passenger` | N:1 | `booking` | `booking_id` | Passengers on a booking |
| `passenger` | N:1 | `account` | `account_id` | Passenger's account (if registered) |
| `booking_leg` | N:1 | `booking` | `booking_id` | Flight segments in booking |
| `booking_leg` | N:1 | `flight` | `flight_id` | Links to actual flight |
| `flight` | N:1 | `aircraft` | `aircraft_code` | Aircraft type for flight |
| `flight` | N:1 | `airport` | `departure_airport` | Origin airport |
| `flight` | N:1 | `airport` | `arrival_airport` | Destination airport |
| `boarding_pass` | N:1 | `passenger` | `passenger_id` | Pass for passenger |
| `boarding_pass` | N:1 | `booking_leg` | `booking_leg_id` | Pass for flight segment |

## Table Statistics

| Table | Primary Key | Foreign Keys | Purpose |
|-------|-------------|--------------|---------|
| `aircraft` | `code` | 0 | Aircraft fleet catalog |
| `airport` | `airport_code` | 0 | Airport master data |
| `frequent_flyer` | `frequent_flyer_id` | 0 | Loyalty program members |
| `account` | `account_id` | 1 | User accounts |
| `phone` | `phone_id` | 1 | Account phone numbers |
| `booking` | `booking_id` | 1 | Flight reservations |
| `passenger` | `passenger_id` | 2 | Travelers on bookings |
| `flight` | `flight_id` | 3 | Scheduled flights |
| `booking_leg` | `booking_leg_id` | 2 | Flight segments in bookings |
| `boarding_pass` | `pass_id` | 2 | Boarding documents |

## Schema Design Notes

1. **Core Entities (No FKs)**: `aircraft`, `airport`, `frequent_flyer` - These are master data tables
2. **Central Hub**: `booking` connects accounts to flights through `booking_leg`
3. **Junction Pattern**: `booking_leg` acts as a junction between `booking` and `flight`
4. **Dual Airport Reference**: `flight` references `airport` twice (departure + arrival)
5. **Optional Relationships**: Many FKs are nullable (e.g., `account_id` in `passenger`)
