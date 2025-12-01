from route_designer.route_designer import design_routes

# -----------------------------------------------------
# TEST SCRIPT FOR RODES
# -----------------------------------------------------
# Run this file from the project root:
#    python test.py
# -----------------------------------------------------

if __name__ == "__main__":
    routes = design_routes(
        trips_csv_path="dataset_generator/london_ridership_trip_generation.csv",
        stops_csv_path="dataset_generator/stopwithZones.csv",
        depot_id="ADELCEN1",    
        vehicle_capacity=60,
        save_path="generated_routes.json"
    )

    print("\nFinal Routes:")
    for i, r in enumerate(routes, 1):
        print(f"Route {i}: {r}")
