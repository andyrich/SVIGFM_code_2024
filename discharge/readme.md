# stream discharge measurements

these are the results from years of stream discharge measurements in Sonoma Valley.

## recommended processing
1. spatial join Q obs to SFR locations
2. carefully determine whether each measurement belongs to stream segment
3. for those that match stream segment, extract seg/reach and stress period.
4. for each unique location in sfr, extract ALL simulated discharge values (to keep in pest)
5. for those records with matching discharge values, add observed value
6. create pest instruction file to extract above data from outputs