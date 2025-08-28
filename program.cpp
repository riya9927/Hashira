#include <bits/stdc++.h>
using namespace std;

const double EPS = 1e-9;

// Decode string number from given base into long long
long long decodeValue(const string &value, int base) {
    return stoll(value, nullptr, base);
}

// Improved Gaussian elimination solver with better precision handling
vector<double> gaussianElimination(vector<vector<double>> A, vector<double> b) {
    int n = b.size();

    // Forward elimination
    for (int p = 0; p < n; p++) {
        // Find pivot
        int maxRow = p;
        for (int i = p + 1; i < n; i++) {
            if (abs(A[i][p]) > abs(A[maxRow][p])) {
                maxRow = i;
            }
        }
        
        // Swap rows
        swap(A[p], A[maxRow]);
        swap(b[p], b[maxRow]);

        // Check for singular matrix
        if (abs(A[p][p]) < EPS) {
            throw runtime_error("Singular matrix!");
        }

        // Make all rows below this one 0 in current column
        for (int i = p + 1; i < n; i++) {
            double factor = A[i][p] / A[p][p];
            b[i] -= factor * b[p];
            for (int j = p; j < n; j++) {
                A[i][j] -= factor * A[p][j];
            }
        }
    }

    // Back substitution
    vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
        
        // Clean up very small values (likely numerical errors)
        if (abs(x[i]) < EPS) {
            x[i] = 0.0;
        }
    }
    return x;
}

// Alternative: Lagrange interpolation for finding f(0) directly
double lagrangeInterpolationAt0(const vector<pair<int, long long>>& points) {
    double result = 0.0;
    int n = points.size();
    
    for (int i = 0; i < n; i++) {
        double term = points[i].second; // y_i
        double numerator = 1.0;
        double denominator = 1.0;
        
        for (int j = 0; j < n; j++) {
            if (i != j) {
                // For f(0): (0 - x_j) / (x_i - x_j) = -x_j / (x_i - x_j)
                numerator *= (0 - points[j].first);
                denominator *= (points[i].first - points[j].first);
            }
        }
        
        term = term * numerator / denominator;
        result += term;
    }
    
    return result;
}

int main() {
    ifstream inFile("input.json");
    if (!inFile) {
        cerr << "Error: could not open input.json\n";
        return 1;
    }

    string line, jsonStr;
    while (getline(inFile, line)) {
        jsonStr += line;
    }
    inFile.close();

    // Remove spaces for easier parsing
    jsonStr.erase(remove_if(jsonStr.begin(), jsonStr.end(), ::isspace), jsonStr.end());

    // Extract n and k
    size_t posN = jsonStr.find("\"n\":");
    size_t posK = jsonStr.find("\"k\":");
    
    if (posN == string::npos || posK == string::npos) {
        cerr << "Error: Could not find n or k in JSON\n";
        return 1;
    }
    
    int n = stoi(jsonStr.substr(posN + 4, jsonStr.find(",", posN) - posN - 4));
    int k = stoi(jsonStr.substr(posK + 4, jsonStr.find("}", posK) - posK - 4));

    cout << "n = " << n << ", k = " << k << "\n\n";

    vector<pair<int, long long>> points;

    // Parse JSON manually to extract points
    size_t pos = 0;
    while (true) {
        // Find next key
        size_t keyStart = jsonStr.find("\"", pos);
        if (keyStart == string::npos) break;

        size_t keyEnd = jsonStr.find("\"", keyStart + 1);
        if (keyEnd == string::npos) break;
        
        string key = jsonStr.substr(keyStart + 1, keyEnd - keyStart - 1);
        pos = keyEnd + 1;

        // Skip "keys" and non-numeric keys
        if (key == "keys" || !all_of(key.begin(), key.end(), ::isdigit)) {
            continue;
        }

        int x = stoi(key);

        // Find base
        size_t basePos = jsonStr.find("\"base\":\"", pos);
        if (basePos == string::npos) break;
        basePos += 8;
        size_t baseEnd = jsonStr.find("\"", basePos);
        if (baseEnd == string::npos) break;
        
        int base = stoi(jsonStr.substr(basePos, baseEnd - basePos));

        // Find value
        size_t valPos = jsonStr.find("\"value\":\"", baseEnd);
        if (valPos == string::npos) break;
        valPos += 9;
        size_t valEnd = jsonStr.find("\"", valPos);
        if (valEnd == string::npos) break;
        
        string value = jsonStr.substr(valPos, valEnd - valPos);

        try {
            long long y = decodeValue(value, base);
            points.push_back({x, y});
            cout << "Decoded point: (" << x << ", " << y << ")\n";
        } catch (const exception& e) {
            cerr << "Error decoding value: " << value << " in base " << base << "\n";
        }

        pos = valEnd + 1;
    }

    if (points.size() < k) {
        cerr << "Error: Not enough points. Need " << k << ", got " << points.size() << "\n";
        return 1;
    }

    // Sort points by x value
    sort(points.begin(), points.end());

    // Use first k points
    vector<pair<int, long long>> selectedPoints(points.begin(), points.begin() + k);

    cout << "\nUsing points for interpolation:\n";
    for (const auto& p : selectedPoints) {
        cout << "(" << p.first << ", " << p.second << ")\n";
    }

    // Method 1: Direct Lagrange interpolation for f(0)
    double secret = lagrangeInterpolationAt0(selectedPoints);
    cout << "\nSecret (f(0) using Lagrange): " << (long long)round(secret) << "\n";

    // Method 2: Gaussian elimination to find all coefficients
    try {
        vector<vector<double>> A(k, vector<double>(k));
        vector<double> b(k);

        // Build system: coefficient of x^(k-1), x^(k-2), ..., x^1, x^0
        for (int i = 0; i < k; i++) {
            double xi = selectedPoints[i].first;
            double yi = selectedPoints[i].second;

            double power = 1.0;
            // Fill from right to left: a0, a1, a2, ..., a(k-1)
            for (int j = k - 1; j >= 0; j--) {
                A[i][j] = power;
                power *= xi;
            }
            b[i] = yi;
        }

        vector<double> coeffs = gaussianElimination(A, b);

        cout << "\nPolynomial coefficients:\n";
        for (int i = 0; i < coeffs.size(); i++) {
            int power = coeffs.size() - 1 - i;
            double coeff = coeffs[i];
            
            // Round very small values to 0
            if (abs(coeff) < EPS) {
                coeff = 0.0;
            }
            
            cout << "a" << power << " = ";
            if (abs(coeff - round(coeff)) < EPS) {
                cout << (long long)round(coeff) << "\n";
            } else {
                cout << coeff << "\n";
            }
        }

        cout << "\nThe secret (constant term a0): " << (long long)round(coeffs[k-1]) << "\n";

    } catch (const exception& e) {
        cerr << "Error in Gaussian elimination: " << e.what() << "\n";
        return 1;
    }

    return 0;
}